#!/usr/local/bin/python3.3

"""This module enriches transactions with additional
data found by Meerkat

Created on Nov 3, 2014
@author: Matthew Sevrens
"""

import json
# pylint:disable=deprecated-module
import string
import re
import logging
import os
from multiprocessing.pool import ThreadPool
from scipy.stats.mstats import zscore

from meerkat.various_tools import get_es_connection, string_cleanse, get_boosted_fields
from meerkat.various_tools import synonyms, get_bool_query, get_qs_query
from meerkat.classification.load_model import load_scikit_model, get_tf_cnn_by_path
from meerkat.classification.auto_load import main_program as load_models_from_s3
from meerkat.various_tools import load_params

class WebConsumerDatadeal():
	"""Acts as a web service client to process and enrich
	transactions in real time"""

	# 14 is the best thread number Andy has tried
	__cpu_pool = ThreadPool(processes=14)

	def __init__(self, params=None, hyperparams=None, cities=None):
		"""Constructor"""

		if params is None:
			self.params = dict()
		else:
			self.params = params
			if not params["elasticsearch"]["skip_es"]:
				self.elastic_search = get_es_connection(params)
				mapping = self.elastic_search.indices.get_mapping()
				index = params["elasticsearch"]["index"]
				index_type = params["elasticsearch"]["type"]
				self.params["routed"] = "_routing" in mapping[index]["mappings"][index_type]

		self.load_merchant_name_map()
		self.load_tf_models()
		self.hyperparams = hyperparams if hyperparams else {}
		self.cities = cities if cities else {}

	def load_merchant_name_map(self):
		"""Load a json map to convert CNN merchant name to Agg merchant names"""
		merchant_name_map_path = self.params.get("merchant_name_map_path", None)
		if merchant_name_map_path is not None:
			self.merchant_name_map = load_params(merchant_name_map_path)

	def load_tf_models(self):
		"""Load all tensorFlow models"""

		gmf = self.params.get("gpu_mem_fraction", False)
		auto_load_config = self.params.get("auto_load_config", None)

		#Auto load cnn models from S2, if necessary
		if auto_load_config is not None:
			#Flush old models, to be safe
			target_dir = "meerkat/classification/models/"
			for file_name in os.listdir(target_dir):
				if file_name.endswith(".meta") or file_name.endswith(".ckpt"):
					file_path = os.path.join(target_dir, file_name)
					logging.warning("Removing {0}".format(file_path))
					os.unlink(file_path)
			#Load new models from S3
			load_models_from_s3(config=auto_load_config)

		# Get CNN Models
		self.models = dict()
		models_dir = 'meerkat/classification/models/'
		label_maps_dir = "meerkat/classification/label_maps/"
		for filename in os.listdir(models_dir):
			if filename.endswith('.ckpt') and not filename.startswith('train'):
				temp = filename.split('.')[:-1]
				if temp[-1][-1].isdigit():
					key = '_'.join(temp[1:-1] + [temp[0], temp[-1], 'cnn'])
				else:
					key = '_'.join(temp[1:] + [temp[0], 'cnn'])
				self.models[key] = get_tf_cnn_by_path(models_dir + filename, \
					label_maps_dir + filename[:-4] + 'json', gpu_mem_fraction=gmf)

	def __apply_merchant_cnn(self, data):
		"""Apply the merchant CNN to transactions"""

		logging.info(data)
		if "cobrand_region" in data:
			region = 'region_' + str(data["cobrand_region"])
		else:
			region = 'default'

		classifier = None
		if data["container"] == "bank":
			if 'bank_merchant_' + region + '_cnn' not in self.models:
				classifier = self.models['bank_merchant_cnn']
			else:
				classifier = self.models['bank_merchant_' + region + '_cnn']
		else:
			if 'card_merchant_' + region + '_cnn' not in self.models:
				classifier = self.models['card_merchant_cnn']
			else:
				classifier = self.models['card_merchant_' + region + '_cnn']
		return classifier(data["transaction_list"], label_only=False)

	def __choose_agg_or_factual(self, data):
		"""Split transactions to search in agg index or factual index"""
		data_to_search_in_agg, data_to_search_in_factual = [], []
		for trans in data["transaction_list"]:
			cnn_merchant = trans['CNN']['label']
			if cnn_merchant != '' and cnn_merchant in self.merchant_name_map:
				trans["agg_merchant_name"] = self.merchant_name_map[cnn_merchant]
				data_to_search_in_agg.append(trans)
			else:
				"""
				if cnn_merchant != '':
					trans["merchant_name"] = cnn_merchant
				else:
					# TODO: get rnn_merchant
					trans["merchant_name"] = rnn_merchant
				"""
				data_to_search_in_factual.append(trans)
		return data_to_search_in_agg, data_to_search_in_factual

	def __search_in_agg_or_factual(self, data):
		for transaction in data["transaction_list"]:
			transaction["container"] = data["container"]
			transaction["datadeal"] = True
		data_to_search_in_agg, data_to_search_in_factual = self.__choose_agg_or_factual(data)
		"""
		search_agg_in_batch = self.params.get("search_agg_in_batch", True)
		search_factual_in_batch = self.params.get("search_factual_in_batch", True)
		if search_agg_in_batch:
			data_to_search_in_agg = self.__enrich_by_agg(data_to_search_in_agg)
		else:
			for i in range(len(data_to_search_in_agg)):
				data_to_search_in_agg[i] = self.__enrich_by_agg(data_to_search_in_agg[i])
		if search_factual_in_batch:
			data_to_search_in_factual = self.__enrich_by_factual(data_to_search_in_factual)
		else:
			for i in range(len(data_to_search_in_factual)):
				data_to_search_in_factual[i] = self.__enrich_by_factual(data_to_search_in_factual[i])
		"""
		return data_to_search_in_agg, data_to_search_in_factual

	def classify(self, data, optimizing=False):
		"""Classify a set of transactions"""

		# Apply Merchant CNN
		self.__apply_merchant_cnn(data)

		# Apply RNN
		#self.__apply_rnn(data)

		# Apply Elasticsearch
		self.__search_in_agg_or_factual(data)

		# Process enriched data to ensure output schema
		#self.ensure_output_schema(data["transaction_list"])

		return data

if __name__ == "__main__":
	# Print a warning to not execute this file as a module
	print("This module is a Class; it should not be run from the console.")

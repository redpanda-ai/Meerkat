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
from meerkat.classification.load_model import (load_scikit_model,
	get_tf_cnn_by_path, get_tf_rnn_by_path)
from meerkat.classification.auto_load import main_program as load_models_from_s3
from meerkat.various_tools import load_params
from meerkat.elasticsearch.search_agg_index import search_agg_index

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
			if (filename.startswith('merchant') and filename.endswith('.ckpt') and
				not filename.startswith('train')):
				temp = filename.split('.')[:-1]
				if temp[-1][-1].isdigit():
					key = '_'.join(temp[1:-1] + [temp[0], temp[-1], 'cnn'])
				else:
					key = '_'.join(temp[1:] + [temp[0], 'cnn'])
				self.models[key] = get_tf_cnn_by_path(models_dir + filename, \
					label_maps_dir + filename[:-4] + 'json', gpu_mem_fraction=gmf)

		# Get RNN Models
		"""
		rnn_model_path = "./meerkat/classification/models/rnn_model/bilstm.ckpt"
		w2i_path = "./meerkat/classification/models/rnn_model/w2i.json"
		if os.path.exists(rnn_model_path) is False:
			logging.warning("Please run python3 -m meerkat.longtail.rnn_auto_load")
		self.models["rnn"] = get_tf_rnn_by_path(rnn_model_path, w2i_path)
		"""

	def __get_query(self, transaction):
		"""Create an optimized query"""

		result_size = self.hyperparams.get("es_result_size", "10")
		fields = self.params["output"]["results"]["fields"]
		city, state = transaction["city"], transaction["state"]
		phone_number, postal_code = transaction["phone_number"], transaction["postal_code"]

		# Construct Optimized Query
		o_query = get_bool_query(size=result_size)
		o_query["fields"] = fields
		o_query["_source"] = "pin.*"
		should_clauses = o_query["query"]["bool"]["should"]
		must_clauses = o_query["query"]["bool"]["must"]
		field_boosts = get_boosted_fields(self.hyperparams, "standard_fields")
		# Add Merchant Sub Query
		if transaction['CNN']['label'] != '':
			term = string_cleanse(transaction['CNN']['label'])
			merchant_query = get_qs_query(term, ['name'], field_boosts['name'], operator="AND")
			must_clauses.append(merchant_query)
		elif transaction['RNN_merchant_name'] != '':
			term = synonyms(transaction['RNN_merchant_name'])
			term = string_cleanse(term)
			merchant_query = get_qs_query(term, ['name'], field_boosts['name'])
			merchant_query["query_string"]["fuzziness"] = "AUTO"
			must_clauses.append(merchant_query)

		# Add Locale Sub Query
		if city != '':
			city_query = get_qs_query(city, ['locality'], field_boosts['locality'], operator="AND")
			city_query["query_string"]["fuzziness"] = "AUTO"
			should_clauses.append(city_query)
		if state != '':
			state_query = get_qs_query(state, ['region'], field_boosts['region'], operator="AND")
			must_clauses.append(state_query)
		if phone_number != '':
			phone_query = get_qs_query(phone_number, ['tel'], field_boosts['tel'], operator="AND")
			should_clauses.append(phone_query)
		if postal_code != '':
			postcode_query = get_qs_query(postal_code, ['postcode'], field_boosts['postcode'], operator="AND")
			must_clauses.append(postcode_query)

		return o_query

	def __search_index(self, queries):
		"""Search against a structured index"""
		index = self.params["elasticsearch"]["index"]
		results = self.elastic_search.msearch(queries, index=index)
		return results

	@staticmethod
	def __z_score_delta(scores):
		"""Find the Z-Score Delta"""
		if len(scores) < 2:
			return None, None
		z_scores = zscore(scores)
		first_score, second_score = z_scores[0:2]
		z_score_delta = round(first_score - second_score, 3)
		return z_score_delta, scores[0]

	def __process_results(self, results, transaction):
		"""Process search results and enrich transaction
		with found data"""

		hyperparams = self.hyperparams

		# Must be at least one result
		if "hits" not in results or results["hits"]["total"] == 0:
			transaction = self.__no_result(transaction)
			return transaction

		# Collect Necessary Information
		hits = results['hits']['hits']
		if len(hits) < 1:
			transaction = self.__no_result(transaction)
			return transaction
		top_hit = hits[0]
		hit_fields = top_hit.get("fields", "")

		# If no results return
		if hit_fields == "":
			transaction = self.__no_result(transaction)
			return transaction

		# Elasticsearch v1.0 bug workaround
		if top_hit["_source"].get("pin", "") != "":
			coordinates = top_hit["_source"]["pin"]["location"]["coordinates"]
			hit_fields["longitude"] = "%.6f" % (float(coordinates[0]))
			hit_fields["latitude"] = "%.6f" % (float(coordinates[1]))

		# Collect Fallback Data
		names = {}
		names["business_names"] =\
			[result.get("fields", {"name" : ""}).get("name", "") for result in hits]
		names["business_names"] =\
			[name[0] for name in names["business_names"] if isinstance(name, list)]
		names["city_names"] =\
			[result.get("fields", {"locality" : ""}).get("locality", "") for result in hits]
		names["city_names"] = [name[0] for name in names["city_names"] if isinstance(name, list)]
		names["state_names"] =\
			[result.get("fields", {"region" : ""}).get("region", "") for result in hits]
		names["state_names"] = [name[0] for name in names["state_names"] if isinstance(name, list)]

		# Need Names
		if len(names["business_names"]) < 2:
			names["business_names"].append("")

		if len(names["city_names"]) < 2:
			names["city_names"].append("")

		if len(names["state_names"]) < 2:
			names["state_names"].append("")

		# Collect Relevancy Scores
		scores = [hit["_score"] for hit in hits]
		thresholds = [float(hyperparams.get("z_score_threshold", "2")), \
			float(hyperparams.get("raw_score_threshold", "1"))]

		if len(scores) == 1:
			z_score_delta, raw_score = thresholds[0] + 1, scores[0]
		elif len(scores) == 2:
			z_score_delta, raw_score = scores[0] - scores[1], scores[0]
		else:
			z_score_delta, raw_score = self.__z_score_delta(scores)
		decision = True if (z_score_delta > thresholds[0]) and (raw_score > thresholds[1]) else False
		
		# Enrich Data if Passes Boundary
		args = [decision, transaction, hit_fields,\
			 names["business_names"], names["city_names"], names["state_names"], z_score_delta]
		return self.__enrich_transaction(*args)

	def __no_result(self, transaction):
		"""Make sure transactions have proper attribute names"""

		params = self.params
		fields = params["output"]["results"]["fields"]
		labels = params["output"]["results"]["labels"]
		attr_map = dict(zip(fields, labels))
		transaction["factual_search"] = dict()

		for field in fields:
			transaction["factual_search"][attr_map.get(field, field)] = ""

		transaction["factual_search"]["match_found"] = False
		# Add fields required
		if transaction["factual_search"]["country"] == "":
			transaction["factual_search"]["country"] = "US"
		transaction["factual_search"]["source"] = "FACTUAL"
		transaction["factual_search"]["confidence_score"] = ""

		return transaction

	def __enrich_transaction(self, *argv):
		"""Enriches the transaction with additional data"""

		decision = argv[0]
		transaction = argv[1]
		hit_fields = argv[2]
		business_names = argv[3]
		city_names = argv[4]
		state_names = argv[5]

		params = self.params
		field_names = params["output"]["results"]["fields"]
		fields_in_hit = [field for field in hit_fields]
		transaction["match_found"] = False

		# Collect Mapping Details
		attr_map = dict(zip(params["output"]["results"]["fields"], params["output"]["results"]["labels"]))
		transaction["factual_search"] = dict()

		# Enrich with found data
		if decision is True:
			transaction["match_found"] = True
			for field in field_names:
				if field in fields_in_hit:
					field_content = hit_fields[field][0] if isinstance(hit_fields[field],\
 						(list)) else str(hit_fields[field])
					transaction["factual_search"][attr_map.get(field, field)] = field_content
				else:
					transaction["factual_search"][attr_map.get(field, field)] = ""

			if not transaction.get("country") or transaction["country"] == "":
				logging.warning(("Factual response for merchant {} has no country code. "
					"Defaulting to US.").format(hit_fields["factual_id"][0]))
				transaction["factual_search"]["country"] = "US"
		# Add Business Name, City and State as a fallback
		if decision is False:
			for field in field_names:
				transaction["factual_search"][attr_map.get(field, field)] = ""
			transaction = self.__business_name_fallback(business_names, transaction, attr_map)
			transaction = self.__geo_fallback(city_names, state_names, transaction, attr_map)
			#Ensuring that there is a country code that matches the schema limitation
			transaction["factual_search"]["country"] = "US"

		# Ensure Proper Casing
		if transaction["factual_search"][attr_map['name']] == transaction["factual_search"][attr_map['name']].upper():
			transaction["factual_search"][attr_map['name']] = string.capwords(transaction["factual_search"][attr_map['name']], " ")
		transaction["factual_search"]["confidence_score"] = ""

		return transaction

	@staticmethod
	def __geo_fallback(city_names, state_names, transaction, attr_map):
		"""Basic logic to obtain a fallback for city and state
		when no factual_id is found"""
		city_names = city_names[0:2]
		state_names = state_names[0:2]
		states_equal = state_names.count(state_names[0]) == len(state_names)
		city_in_transaction = (city_names[0].lower() in transaction["description"].lower())
		state_in_transaction = (state_names[0].lower() in transaction["description"].lower())

		if city_in_transaction:
			transaction[attr_map['locality']] = city_names[0]

		if states_equal and state_in_transaction:
			transaction[attr_map['region']] = state_names[0]

		return transaction

	def __business_name_fallback(self, business_names, transaction, attr_map):
		"""Basic logic to obtain a fallback for business name
		when no factual_id is found"""
		business_names = business_names[0:2]
		top_name = business_names[0].lower()
		all_equal = business_names.count(business_names[0]) == len(business_names)
		not_a_city = top_name not in self.cities

		if all_equal and not_a_city:
			transaction[attr_map['name']] = business_names[0]

		return transaction

	def __search_factual_index(self, transactions):
		"""Enrich physical transactions with Meerkat"""
		if len(transactions) == 0:
			return transactions

		enriched, queries = [], []
		index = self.params["elasticsearch"]["index"]

		for trans in transactions:
			query = self.__get_query(trans)

			header = {"index": index}
			# add routing to header
			if self.params["routed"] and trans.get("locale_bloom", None):
				region = trans["locale_bloom"][1]
				header["routing"] = region.upper()

			queries.append(header)
			queries.append(query)

		queries = '\n'.join(map(json.dumps, queries))
		results = self.__search_index(queries)

		# Error Handling
		if results is None:
			return transactions

		results = results['responses']

		for result, transaction in zip(results, transactions):
			trans_plus = self.__process_results(result, transaction)
			enriched.append(trans_plus)

		return enriched

	def __apply_rnn(self, trans):
		"""Apply RNN to transactions"""
		classifier = self.models["rnn"]
		return classifier(trans, doc_key="description", label_key="RNN_merchant_name")

	def __apply_merchant_cnn(self, data):
		"""Apply the merchant CNN to transactions"""

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
			if trans.get("country", "") not in ["", "US", "USA"]:
				continue
			cnn_merchant = trans['CNN']['label']
			if cnn_merchant != '' and cnn_merchant in self.merchant_name_map:
				trans["Agg_Name"] = self.merchant_name_map[cnn_merchant]
				data_to_search_in_agg.append(trans)
			else:
				data_to_search_in_factual.append(trans)
		return data_to_search_in_agg, data_to_search_in_factual

	def __search_in_agg_or_factual(self, data):
		"""Enrich transactions with agg search or factual search"""
		for transaction in data["transaction_list"]:
			transaction["container"] = data["container"]
		data_to_search_in_agg, data_to_search_in_factual = self.__choose_agg_or_factual(data)
		search_agg_in_batch = self.params.get("search_agg_in_batches", True)
		search_factual_in_batch = self.params.get("search_factual_in_batches", True)
		if search_agg_in_batch:
			data_to_search_in_agg = search_agg_index(data_to_search_in_agg)
		else:
			for i in range(len(data_to_search_in_agg)):
				data_to_search_in_agg[i] = search_agg_index([data_to_search_in_agg[i]])
		if search_factual_in_batch:
			data_to_search_in_factual = self.__search_factual_index(data_to_search_in_factual)
		else:
			for i in range(len(data_to_search_in_factual)):
				data_to_search_in_factual[i] = self.__search_factual_index([data_to_search_in_factual[i]])
		return data_to_search_in_agg, data_to_search_in_factual

	def enrich_with_search_fields(self, trans, agg_or_factual, map_input_fields, fields_with_same_name):
		"""Enrich transaction with fields in search"""
		# Override input fields by search
		for field in map_input_fields:
			search_field = map_input_fields[field]
			if trans[agg_or_factual].get(search_field, "") != "":
				trans[field] = trans[agg_or_factual][search_field]

		# Add search fields to transaction
		for field in fields_with_same_name:
			if trans[agg_or_factual].get(field, "") != "":
				trans[field] = trans[agg_or_factual].get(field, "")

	def ensure_output_schema(self, transactions, debug):
		"""Merge fields and clean output to proper schema"""
		for trans in transactions:
			# In debug mode, keep all input fields
			if debug:
				input_fields = ["city", "state", "country", "postal_code", "RNN_merchant_name",
					"store_number", "phone_number", "website_url"]
				trans["input"] = {}
				for field in input_fields:
					trans["input"][field] = trans.get(field, "")

			# Enrich transaction with merchant name
			if trans.get("CNN", "") != "" and trans["CNN"].get("label", "") != '':
				trans["merchant_name"] = trans["CNN"]["label"]
			elif trans.get("agg_search", "") != "" and trans["agg_search"].get("list_name") != "":
				trans["merchant_name"] = trans["agg_search"]["list_name"]
			elif trans.get("factual_search", "") != "" and trans["factual_search"].get("merchant_name") != "":
				trans["merchant_name"] = trans["factual_search"]["merchant_name"]
			else:
				trans["merchant_name"] = trans["RNN_merchant_name"]

			# Enrich transaction with fields found in search
			if "agg_search" in trans:
				same_name_for_agg_and_input = ["city", "state", "phone_number", "longitude",
					"latitude", "store_number", "address"]
				map_input_fields_to_agg = {
					"postal_code": "zip_code",
					"website_url": "source_url"
				}
				self.enrich_with_search_fields(trans, "agg_search",
					map_input_fields_to_agg, same_name_for_agg_and_input)
			elif "factual_search" in trans:
				same_name_for_factual_and_input = ["city", "state", "phone_number", "longitude",
					"latitude", "postal_code"]
				map_input_fields_to_factual = {
					"address": "street",
					"website_url": "website"
				}
				self.enrich_with_search_fields(trans, "factual_search",
					map_input_fields_to_factual, same_name_for_factual_and_input)

			# Ensure these fields exist in output
			output_fields = ["city", "state", "address", "longitude", "latitude",
				"website_url", "store_number", "phone_number", "postal_code",
				"transaction_id"]
			map_fields_for_output = {
				"phone_number": "phone",
				"postal_code": "zip_code",
				"transaction_id": "row_id"
			}
			for field in output_fields:
				if field in map_fields_for_output:
					output_field = map_fields_for_output[field]
					trans[output_field] = trans.get(field, "")
					trans.pop(field, None)
				else:
					trans[field] = trans.get(field, "")

			# Remove fields not in output schema
			if debug is False:
				fields_to_remove = ["description", "amount", "date", "ledger_entry", "CNN",
					"container", "RNN_merchant_name", "Agg_Name", "factual_search",
					"agg_search", "merchant_score", "country", "match_found"]
				for field in fields_to_remove:
					trans.pop(field, None)
			else:
				trans['CNN']['merchant_score'] =  trans.get("merchant_score", "0.0")
				trans['CNN'].pop("threshold", None)
				trans['CNN'].pop("category", None)
				fields_to_remove = ["amount", "date", "ledger_entry", "container",
					"merchant_score", "country", "match_found"]
				for field in fields_to_remove:
					trans.pop(field, None)

	def classify(self, data, optimizing=False):
		"""Classify a set of transactions"""
		debug = data.get("debug", False)

		# Apply Merchant CNN
		self.__apply_merchant_cnn(data)

		# Apply Elasticsearch
		self.__search_in_agg_or_factual(data)

		# Process enriched data to ensure output schema
		self.ensure_output_schema(data["transaction_list"], debug)

		return data

if __name__ == "__main__":
	# Print a warning to not execute this file as a module
	print("This module is a Class; it should not be run from the console.")

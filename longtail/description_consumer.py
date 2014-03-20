'''
Created on Jan 14, 2014

@author: J. Andrew Key
@author: Matthew Sevrens
'''

#!/bin/python3
# pylint: disable=R0914

import copy
import hashlib
import json
import logging
import numpy as np
import pickle
import queue
import re
import sys
import pprint
import datetime
import threading

from elasticsearch import Elasticsearch, helpers
from scipy.stats.mstats import zscore

from longtail.custom_exceptions import Misconfiguration, UnsupportedQueryType
from longtail.various_tools import string_cleanse

#Helper functions
def get_bool_query(starting_from = 0, size = 0):
	"""Returns a "bool" style ElasticSearch query object"""
	return { "from" : starting_from, "size" : size, "query" : {
		"bool": { "minimum_number_should_match": 1, "should": [] } } }

def get_basic_query(starting_from=0, size=0):
	"""Returns an ElasticSearch query object"""
	return {"from" : starting_from, "size" : size, "query" : {}}

def get_qs_query(term, field_list=[], boost=1.0):
	"""Returns a "query_string" style ElasticSearch query object"""
	return { "query_string": {
		"query": term, "fields": field_list, "boost": boost } }

def get_fuzzy_query(term, field_list=["_all"]):
	"""Returns a "fuzzy_like_this" style ElasticSearch query object"""
	return { "fuzzy_like_this": {
		"like_text": term, "fields": field_list}}

class DescriptionConsumer(threading.Thread):
	''' Acts as a client to an ElasticSearch cluster, tokenizing description
	strings that it pulls from a synchronized queue. '''

	def __build_boost_vectors(self):
		"""Turns configuration entries into a dictionary of numpy arrays."""
		logger = logging.getLogger("thread " + str(self.thread_id))
		boost_column_labels = self.params["elasticsearch"]["boost_labels"]
		boost_row_vectors = self.params["elasticsearch"]["boost_vectors"]
		boost_row_labels, boost_column_vectors = sorted(boost_row_vectors.keys()), {}
		for i in range(len(boost_column_labels)):
			my_list = []
			for field in boost_row_labels:
				my_list.append(boost_row_vectors[field][i])
			boost_column_vectors[boost_column_labels[i]] = np.array(my_list)
		return boost_row_labels, boost_column_vectors

	def __display_search_results(self, search_results):
		"""Displays search results."""

		# Must have results
		if search_results['hits']['total'] == 0:
			return True	

		hits = search_results['hits']['hits']
		scores, results, fields_found = [], [], []
		params = self.params
		for hit in hits:
			hit_fields, score = hit['fields'], hit['_score']
			scores.append(score)
			field_order = params["output"]["results"]["fields"]
			fields_in_hit = [field for field in hit_fields]
			ordered_hit_fields = []
			for ordinal in field_order:
				if ordinal in fields_in_hit:
					my_field = hit_fields[ordinal][0] if isinstance(hit_fields[ordinal], (list)) else str(hit_fields[ordinal])
					fields_found.append(ordinal)
					ordered_hit_fields.append(my_field)
				else:
					fields_found.append(ordinal)
					ordered_hit_fields.append("")
			results.append(\
			"[" + str(round(score, 3)) + "] " + " ".join(ordered_hit_fields))

		self.__display_z_score_delta(scores)

		try:
			print(str(self.thread_id), ": ", results[0])
		except IndexError:
			print("INDEX ERROR: ", self.input_string)

		return True

	def __display_z_score_delta(self, scores):
		"""Display the Z-score delta between the first and second scores."""
		logger = logging.getLogger("thread " + str(self.thread_id))
		if len(scores) < 2:
			logger.info("Unable to generate Z-Score")
			return None

		z_scores = zscore(scores)
		first_score, second_score = z_scores[0:2]
		z_score_delta = round(first_score - second_score, 3)
		logger.info("Z-Score delta: [%.2f]", z_score_delta)
		quality = "Non"
		if z_score_delta <= 1:
			quality = "Low-grade"
		elif z_score_delta <= 2:
			quality = "Mid-grade"
		else:
			quality = "High-grade"

		logger.info("Top Score Quality: %s", quality)
		return z_score_delta

	def __init__(self, thread_id, params, desc_queue, result_queue, hyperparameters):
		''' Constructor '''
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.desc_queue = desc_queue
		self.result_queue = result_queue
		self.input_string = None
		self.params = params
		self.hyperparameters = hyperparameters

		cluster_nodes = self.params["elasticsearch"]["cluster_nodes"]
		self.es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,
			sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)

		self.my_meta = None
		self.__reset_my_meta()
		self.__set_logger()
		self.boost_row_labels, self.boost_column_vectors =\
			self.__build_boost_vectors()

	def __generate_final_query(self):
		"""Constructs a simple query along with boost vectors"""

		logger = logging.getLogger("thread " + str(self.thread_id))
		hyperparameters = self.hyperparameters
		params = self.params
		result_size = self.hyperparameters.get("es_result_size", "10")
		fuzziness = self.hyperparameters.get("fuzziness", 0.5)
		fields = params["output"]["results"]["fields"]

		# Construct Main Query
		logger.info("BUILDING FINAL BOOLEAN SEARCH")
		bool_search = get_bool_query(size=result_size)
		bool_search["fields"] = fields

		# Ensure we get location
		if "pin.location" not in fields:
			params["output"]["results"]["fields"].append("pin.location")

		should_clauses = bool_search["query"]["bool"]["should"]
		field_boosts = self.__get_boosted_fields("standard_fields")
		input_string = string_cleanse(self.input_string)

		# String must not be empty
		if len(input_string.rstrip()) <= 2:
			return

		# Collect Query Parts
		simple_query = get_qs_query(input_string, field_boosts)
		should_clauses.append(simple_query)

		# Show Final Query
		logger.info(json.dumps(bool_search, sort_keys=True, indent=4, separators=(',', ': ')))
		my_results = self.__search_index(bool_search)
		metrics = self.my_meta["metrics"]
		logger.info("Cache Hit / Miss: %i / %i", metrics["cache_count"], metrics["query_count"])
		self.__display_search_results(my_results)
		self.__output_to_result_queue(my_results)

	def __output_to_result_queue(self, search_results):
		"""Decides whether to output and pushes to result_queue"""

		# Must be at least one result
		if search_results['hits']['total'] == 0:
			field_names = self.params["output"]["results"]["fields"]
			output_dict = dict(zip(field_names, ([""] * len(field_names))))
			output_dict["DESCRIPTION"] = self.input_string
			self.result_queue.put(output_dict)
			print("NO RESULTS: ", self.input_string)
			return

		hits = search_results['hits']['hits']
		scores, fields_found = [], []
		output_dict = {}
		params = self.params
		hyperparameters = self.hyperparameters
		field_order = params["output"]["results"]["fields"]
		top_hit = hits[0]
		hit_fields = top_hit["fields"]
		fields_in_hit = [field for field in hit_fields]
		business_names = [result["fields"]["name"] for result in hits]
		ordered_hit_fields = []

		for hit in hits:
			scores.append(hit['_score'])

		for ordinal in field_order:
			if ordinal in fields_in_hit:
				my_field = hit_fields[ordinal][0] if isinstance(hit_fields[ordinal], (list)) else str(hit_fields[ordinal])
				fields_found.append(ordinal)
				ordered_hit_fields.append(my_field)
			else:
				fields_found.append(ordinal)
				ordered_hit_fields.append("")

		z_score_delta = self.__display_z_score_delta(scores)
		top_score = top_hit['_score']

		#Unable to generate z_score
		if z_score_delta == None:
			output_dict = dict(zip(fields_found, ordered_hit_fields))
			output_dict['DESCRIPTION'] = self.input_string
			self.result_queue.put(output_dict)
			return

		#Send to result Queue if score good enough
		if z_score_delta > float(hyperparameters.get("z_score_threshold", "2")):
			output_dict = dict(zip(fields_found, ordered_hit_fields))
		else:
			output_dict = self.__business_name_fallback(business_names)

		output_dict['DESCRIPTION'] = self.input_string
		self.result_queue.put(output_dict)

		logging.info("Z_SCORE_DELTA: %.2f", z_score_delta)
		logging.info("TOP_SCORE: %.4f", top_score)

	def __business_name_fallback(self, business_names):
		"""Uses the business names as a fallback
		to finding a specific factual id"""

		fields = self.params["output"]["results"]["fields"]
		input_string = self.input_string
		output_dict = dict(zip(fields, ([""] * len(fields))))
		business_names = business_names[0:2]
		all_equal = business_names.count(business_names[0]) == len(business_names)

		if all_equal:
			output_dict['name'] = business_names[0]
		
		return output_dict

	def __reset_my_meta(self):
		"""Purges several object data structures and re-initializes them."""
		self.my_meta = {"metrics": {"query_count": 0, "cache_count": 0}}

	def __search_index(self, input_as_object):
		"""Searches the merchants index and the merchant mapping"""

		input_data = json.dumps(input_as_object, sort_keys=True, indent=4\
		, separators=(',', ': ')).encode('UTF-8')
		output_data = ""
		logger = logging.getLogger("thread " + str(self.thread_id))
		num_transactions = self.result_queue.qsize() + self.desc_queue.qsize()
		use_cache = self.params["elasticsearch"].get("cache_results", True)
		
		if use_cache == True:
			output_data = self.__check_cache(input_data)

		if output_data == "":
			logger.debug("Cache miss, searching")
			try:
				output_data = self.es_connection.search(
					index=self.params["elasticsearch"]["index"], body=input_as_object)
				#Add newly found results to the client cache
				if use_cache == True:
					self.params["search_cache"][input_hash] = output_data
			except Exception:
				logging.critical("Unable to process the following: %s", str(input_as_object))
				output_data = '{"hits":{"total":0}}'

		self.my_meta["metrics"]["query_count"] += 1
		return output_data

	def __check_cache(input_data):

		# Check cache, then run if query is not found
		hash_object = hashlib.md5(str(input_data).encode())
		input_hash = hash_object.hexdigest()

		if input_hash in self.params["search_cache"]:
			logger.debug("Cache hit, short-cutting")
			sys.stdout.write("*")
			sys.stdout.flush()
			self.my_meta["metrics"]["cache_count"] += 1
			output_data = self.params["search_cache"][input_hash]

		return output_data

	def __get_boosted_fields(self, vector_name):
		"""Returns a list of boosted fields built from a boost vector"""
		boost_vector = self.boost_column_vectors[vector_name]
		fields = [ x + "^" + str(y) for x, y in zip(self.boost_row_labels, boost_vector) if y != 0.0 ]
		return fields

	def __set_logger(self):
		"""Creates a logger, based upon the supplied config object."""
		levels = {
			'debug': logging.DEBUG, 'info': logging.INFO,
			'warning': logging.WARNING, 'error': logging.ERROR,
			'critical': logging.CRITICAL
		}
		params = self.params
		my_level = params["logging"]["level"]
		if my_level in levels:
			my_level = levels[my_level]
		my_path = params["logging"]["path"]
		my_formatter = logging.Formatter(params['logging']['formatter'])
		#You'll want to add something to identify the thread
		my_logger = logging.getLogger("thread " + str(self.thread_id))
		my_logger.setLevel(my_level)
		file_handler = logging.FileHandler(my_path)
		file_handler.setLevel(my_level)
		file_handler.setFormatter(my_formatter)
		my_logger.addHandler(file_handler)

		#Add console logging, if configured
		my_console = params["logging"]["console"]
		if my_console is True:
			console_handler = logging.StreamHandler()
			console_handler.setLevel(my_level)
			console_handler.setFormatter(my_formatter)
			my_logger.addHandler(console_handler)

		my_logger.info("Log initialized.")
		params_json = json.dumps(params, sort_keys=True, indent=4\
		, separators=(',', ': '))
		my_logger.debug(params_json)

	def run(self):
		while self.desc_queue.qsize() > 0:
			try:
				self.input_string = self.desc_queue.get()
				self.__generate_final_query()
				self.__reset_my_meta()
				self.desc_queue.task_done()

			except queue.Empty:
				print(str(self.thread_id), " found empty queue, terminating.")

		return True

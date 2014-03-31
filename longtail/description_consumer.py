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
import itertools
import multiprocessing

from elasticsearch import Elasticsearch, helpers
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import zscore
from pprint import pprint

from longtail.custom_exceptions import Misconfiguration, UnsupportedQueryType
from longtail.various_tools import string_cleanse
from longtail.scaled_polygon_test import scale_polygon
from longtail.clustering import cluster, convex_hull, collect_clusters
from longtail.location import separate_geo, visualize

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
		"query": term, "fields": field_list, "boost" : boost} }

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

	def __display_search_results(self, search_results, transaction):
		"""Displays search results."""

		# Must have results
		if search_results['hits']['total'] == 0:
			return True

		hits = search_results['hits']['hits']
		scores, results, fields_found = [], [], []
		params = self.params
		for hit in hits:
			hit_fields, score = hit.get("fields", {}), hit['_score']
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

		self.__generate_z_score_delta(scores)

		try:
			print(str(self.thread_id), ": ", results[0])
		except IndexError:
			print("INDEX ERROR: ", transaction["DESCRIPTION"])

		return True

	def __generate_z_score_delta(self, scores):
		"""Display the Z-score delta between the first and second scores."""

		logger = logging.getLogger("thread " + str(self.thread_id))

		if len(scores) < 2:
			logger.info("Unable to generate Z-Score")
			return None

		z_scores = zscore(scores)
		first_score, second_score = z_scores[0:2]
		z_score_delta = round(first_score - second_score, 3)
		logger.info("Z-Score delta: [%.2f]", z_score_delta)
		quality = ""

		if z_score_delta <= 1:
			quality = "Low-grade"
		elif z_score_delta <= 2:
			quality = "Mid-grade"
		else:
			quality = "High-grade"

		logger.info("Top Score Quality: %s", quality)

		return z_score_delta


	def __decision_boundary(self, z_score_delta):
		"""Decides whether or not we will label transaction
		by factual_id"""

		threshold = self.hyperparameters.get("z_score_threshold", "2")

		if z_score_delta > float(threshold):
			return True
		else:
			return False

	def __init__(self, thread_id, params, desc_queue, result_queue, hyperparameters):
		''' Constructor '''
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.desc_queue = desc_queue
		self.result_queue = result_queue
		self.user = None
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

	def __first_pass(self):
		"""Classify transactions using text features only"""

		enriched_transactions = []

		# Call the Classifier on each transaction and return best results
		while len(self.user) > 0:
			transaction = self.user.pop()
			base_query = self.__generate_base_query(transaction)
			search_results = self.__run_classifier(base_query)
			self.__display_search_results(search_results, transaction)
			enriched_transaction = self.__process_results(search_results, transaction)
			enriched_transactions.append(enriched_transaction)

		return enriched_transactions

	def __second_pass(self, first_pass_results):
		"""Classify transactions using geo and text features"""

		user_id = first_pass_results[0]['MEM_ID']
		hits, non_hits = separate_geo(first_pass_results)
		locations_found = [str(json.loads(hit["pin.location"].replace("'", '"'))["coordinates"]) for hit in hits]
		unique_locations = set(locations_found)
		unique_locations = [json.loads(location.replace("'", '"')) for location in unique_locations]
		enriched_transactions = []		

		# Locate user
		scaled_geoshapes = self.__locate_user(unique_locations, user_id)

		# Use first pass results if no location found
		if not bool(scaled_geoshapes):
			return first_pass_results

		# Create Query
		geo_query = self.__generate_geo_query(scaled_geoshapes)
		
		# Run transactions again with geo_query
		for transaction in non_hits:
			base_query = self.__generate_base_query(transaction)
			should_clauses = base_query["query"]["bool"]["should"]
			should_clauses.append(geo_query)
			search_results = self.__run_classifier(json.dumps(base_query))
			self.__display_search_results(search_results, transaction)
			enriched_transaction = self.__process_results(search_results, transaction)
			enriched_transactions.append(enriched_transaction)

		second_pass_results = hits + enriched_transactions
			
		return second_pass_results

	def __locate_user(self, unique_locations, user_id):
		"""Uses first pass results, to find an approximation
		of a users spending area. Returns estimation as
		a bounded polygon"""

		scaling_factor = self.hyperparameters.get("scaling_factor", "1")
		scaling_factor = float(scaling_factor)
		scaled_geoshapes = None

		# Get Scaled Geoshapes
		if len(unique_locations) >= 3:

			# Cluster location and return geoshapes bounding clusters
			original_geoshapes = cluster(unique_locations)

			# If no clusters are found just use the points themselves
			if len(original_geoshapes) == 0:
				scaled_points = StandardScaler().fit_transform(unique_locations)
				labels = [0 for i in range(len(unique_locations))]
				labels = np.array(labels)
				original_geoshapes = collect_clusters(scaled_points, labels, unique_locations)

			# Scale generated geo shapes
			scaled_geoshapes = [scale_polygon(geoshape, scale=scaling_factor)[1] for geoshape in original_geoshapes]
			
			# Save interesting outputs needs to run in it's own process
			if len(unique_locations) >= 3:
				pool = multiprocessing.Pool()
				arguments = [(unique_locations, original_geoshapes, scaled_geoshapes, user_id)]
				pool.starmap(visualize, arguments)

		return scaled_geoshapes

	def __run_classifier(self, query):
		"""Runs the classifier"""

		# Show Final Query
		logger = logging.getLogger("thread " + str(self.thread_id))
		logger.info(json.dumps(query, sort_keys=True, indent=4, separators=(',', ': ')))
		my_results = self.__search_index(query)
		metrics = self.my_meta["metrics"]
		logger.info("Cache Hit / Miss: %i / %i", metrics["cache_count"], metrics["query_count"])

		return my_results

	def __generate_geo_query(self, scaled_shapes):
		"""Generate multipolygon query for use with"""

		geo = {
			"geo_shape" : {
				"pin.location" : {
					"shape" : {
						"type" : "multipolygon",
						"coordinates": [[scaled_shape] for scaled_shape in scaled_shapes]
					}			
				}
			}
		}

		return geo

	def __generate_base_query(self, transaction):
		"""Generates the basic final query used for both
		the first and second passes"""

		# Collect necessary meta info to generate query
		logger = logging.getLogger("thread " + str(self.thread_id))
		hyperparameters = self.hyperparameters
		params = self.params
		result_size = self.hyperparameters.get("es_result_size", "10")
		fields = params["output"]["results"]["fields"]
		transaction = string_cleanse(transaction["DESCRIPTION"]).rstrip()

		# If we're using masked data, remove anything with 3 X's or more
		transaction = re.sub("X{3,}", "", transaction)

		# Input transaction must not be empty
		if len(transaction) <= 2 and re.match('^[a-zA-Z0-9_]+$', transaction):
			return

		# Ensure we get mandatory fields
		mandatory_fields = ["pin.location", "name"]

		for field in mandatory_fields:
			if field not in fields:
				fields.append(field)

		# Construct Main Query
		logger.info("BUILDING FINAL BOOLEAN SEARCH")
		bool_search = get_bool_query(size=result_size)
		bool_search["fields"] = fields
		should_clauses = bool_search["query"]["bool"]["should"]
		field_boosts = self.__get_boosted_fields("standard_fields")
		simple_query = get_qs_query(transaction, field_boosts)
		should_clauses.append(simple_query)

		return bool_search

	def __process_results(self, search_results, transaction):
		"""Prepare results for decision boundary"""

		field_names = self.params["output"]["results"]["fields"]
		params = self.params
		hyperparameters = self.hyperparameters

		# Must be at least one result
		if search_results["hits"]["total"] == 0:

			for field in field_names:
				transaction[field] = ""
				transaction["z_score_delta"] = 0

			return transaction

		# Collect Necessary Information
		hits = search_results['hits']['hits']
		scores, fields_found = [], []
		output_dict = transaction
		top_hit = hits[0]
		hit_fields = top_hit.get("fields", {})
		business_names = [result.get("fields", {"name" : ""})["name"] for result in hits]
		ordered_hit_fields = []

		# Collect Relevancy Scores
		for hit in hits:
			scores.append(hit['_score'])

		z_score_delta = self.__generate_z_score_delta(scores)
		top_score = top_hit['_score']
		decision = self.__decision_boundary(z_score_delta)

		# Enrich Data if Passes Boundary
		enriched_transaction = self.__enrich_transaction(decision, transaction, hit_fields, z_score_delta, business_names)

		return enriched_transaction

	def __enrich_transaction(self, decision, transaction, hit_fields, z_score_delta, business_names):
		"""Enriches the transaction if it passes the boundary"""

		enriched_transaction = transaction
		field_names = self.params["output"]["results"]["fields"]
		fields_in_hit = [field for field in hit_fields]

		# Enrich with the fields we've found. Attach the z_score_delta
		if decision == True: 
			for field in field_names:
				if field in fields_in_hit:
					field_content = hit_fields[field][0] if isinstance(hit_fields[field], (list)) else str(hit_fields[field])
					enriched_transaction[field] = field_content
				else:
					enriched_transaction[field] = ""
			enriched_transaction["z_score_delta"] = z_score_delta

		# Add a Business Name as a fall back
		if decision == False:
			for field in field_names:
				enriched_transaction[field] = ""
			enriched_transaction = self.__business_name_fallback(business_names, transaction)
			enriched_transaction["z_score_delta"] = 0

		return enriched_transaction

	def __output_to_result_queue(self, enriched_transactions):
		"""Pushes results to the result queue"""

		for transaction in enriched_transactions:
			self.result_queue.put(transaction)

	def __business_name_fallback(self, business_names, transaction):
		"""Uses the business names as a fallback
		to finding a specific factual id"""

		fields = self.params["output"]["results"]["fields"]
		enriched_transaction = transaction
		business_names = business_names[0:2]
		all_equal = business_names.count(business_names[0]) == len(business_names)

		if all_equal:
			enriched_transaction['name'] = business_names[0]
		
		return enriched_transaction

	def __reset_my_meta(self):
		"""Purges several object data structures and re-initializes them."""
		self.my_meta = {"metrics": {"query_count": 0, "cache_count": 0}}

	def __search_index(self, input_as_object):
		"""Searches the merchants index and the merchant mapping"""

		logger = logging.getLogger("thread " + str(self.thread_id))
		use_cache = self.params["elasticsearch"].get("cache_results", True)
		input_data = json.dumps(input_as_object, sort_keys=True, indent=4\
		, separators=(',', ': ')).encode('UTF-8')
		output_data = ""

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
				output_data = {"hits":{"total":0}}

		self.my_meta["metrics"]["query_count"] += 1

		return output_data

	def __check_cache(self, input_data):

		# Check cache, then run if query is not found
		hash_object = hashlib.md5(str(input_data).encode())
		input_hash = hash_object.hexdigest()
		output_data = ""

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
				# Select all transactions from user
				self.user = self.desc_queue.get()

				# Classify using text features only 
				results = self.__first_pass()

				# Classify using text and geo features obtained during first pass
				enriched_transactions = self.__second_pass(results)
				#enriched_transactions = results

				# Output Results to Result Queue
				self.__output_to_result_queue(enriched_transactions)

				# Done
				self.desc_queue.task_done()

			except queue.Empty:
				print(str(self.thread_id), " found empty queue, terminating.")

		return True

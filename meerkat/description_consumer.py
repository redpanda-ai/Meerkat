'''
Created on Jan 14, 2014

@author: J. Andrew Key
@author: Matthew Sevrens
'''

#!/bin/python3

import hashlib
import json
import logging
import numpy as np
import pprint
import queue
import re
import sys
import threading

from elasticsearch import Elasticsearch
from pprint import pprint
from scipy.stats.mstats import zscore
from sklearn.preprocessing import StandardScaler

from .clustering import cluster, collect_clusters
from .decorators import arguments
from .location import separate_geo
from .various_tools import string_cleanse, scale_polygon

#Helper functions
def get_basic_query(starting_from=0, size=0):
	"""Returns an ElasticSearch query object"""
	return {"from" : starting_from, "size" : size, "query" : {}}

def get_bool_query(starting_from=0, size=0):
	"""Returns a "bool" style ElasticSearch query object"""
	return {"from" : starting_from, "size" : size, "query" : { \
		"bool": {"minimum_number_should_match": 1, "should": []}}}

def get_geo_query(scaled_shapes):
	"""Generate multipolygon query for use with"""
	return {"geo_shape" : {"pin.location" : {"shape" : { \
		"type" : "multipolygon", \
		"coordinates": [[scaled_shape] for scaled_shape in scaled_shapes] \
		}}}}

def get_qs_query(term, field_list=None, boost=1.0):
	"""Returns a "query_string" style ElasticSearch query object"""
	if field_list is None:
		field_list = []
	return {"query_string": {"query": term, "fields": field_list, \
		"boost" : boost}}

def get_us_cities():
	"""Load an array of US cities"""
	with open("data/misc/US_Cities.txt") as city_file:
		cities = city_file.readlines()
	cities = [city.lower().rstrip('\n') for city in cities]
	return cities

class DescriptionConsumer(threading.Thread):
	''' Acts as a client to an ElasticSearch cluster, tokenizing description
	strings that it pulls from a synchronized queue. '''

	def __build_boost_vectors(self):
		"""Turns configuration entries into a dictionary of numpy arrays."""
		#logger = logging.getLogger("thread " + str(self.init_args["thread_id"]))
		boost_column_labels = self.init_args["params"]\
			["elasticsearch"]["boost_labels"]
		boost_row_vectors = self.init_args["params"]\
			["elasticsearch"]["boost_vectors"]
		boost_row_labels, boost_column_vectors = sorted(boost_row_vectors.keys()), {}
		for i in range(len(boost_column_labels)):
			my_list = []
			for field in boost_row_labels:
				my_list.append(boost_row_vectors[field][i])
			boost_column_vectors[boost_column_labels[i]] = np.array(my_list)
		return boost_row_labels, boost_column_vectors

	def __business_name_fallback(self, business_names, transaction):
		"""Uses the business names as a fallback
		to finding a specific factual id"""

		enriched_transaction = transaction
		business_names = business_names[0:2]
		top_name = business_names[0].lower()
		all_equal = business_names.count(business_names[0]) == len(business_names)
		name_in_transaction =\
			business_names[0].lower() in transaction["DESCRIPTION"].lower()
		not_a_city = top_name not in self.cities

		if (all_equal and not_a_city) or (name_in_transaction and not_a_city):
			enriched_transaction['name'] = business_names[0]
		return enriched_transaction

	def __check_cache(self, input_data):
		"""Check cache, then run if query is not found"""
		my_logger = logging.getLogger("thread " + str(self.init_args["thread_id"]))
		hash_object = hashlib.md5(str(input_data).encode())
		input_hash = hash_object.hexdigest()
		output_data = ""

		if input_hash in self.init_args["params"]["search_cache"]:
			my_logger.debug("Cache hit, short-cutting")
			sys.stdout.write("*")
			sys.stdout.flush()
			self.my_meta["metrics"]["cache_count"] += 1
			output_data = self.init_args["params"]["search_cache"][input_hash]
		return output_data

	def __decision_boundary(self, z_score_delta):
		"""Decides whether or not we will label transaction
		by factual_id"""

		threshold = self.init_args["hyperparameters"].get("z_score_threshold", "2")

		if z_score_delta is None:
			return False
		if z_score_delta > float(threshold):
			return True
		else:
			return False

	def __display_search_results(self, search_results, transaction):
		"""Displays search results."""

		# Must have results
		if search_results['hits']['total'] == 0:
			return True

		hits = search_results['hits']['hits']
		#hit_meta holds a dictionary of meta information about the hits
		hit_meta = {"scores" : [], "results" : [], "fields_found" : []}
		params = self.init_args["params"]

		for hit in hits:
			# Add Latitude and Longitude
			if hit.get("_source", {}).get("pin", "") != "":
				coordinates = hit["_source"]["pin"]["location"]["coordinates"]
				if "fields" in hit:
					hit["fields"]["longitude"] = [coordinates[0]]
					hit["fields"]["latitude"] = [coordinates[1]]
			hit_fields, score = hit.get("fields", {}), hit['_score']
			hit_meta["scores"].append(score)
			field_order = params["output"]["results"]["fields"]
			fields_in_hit = [field for field in hit_fields]
			ordered_hit_fields = []
			for ordinal in field_order:
				if ordinal in fields_in_hit:
					my_field = (hit_fields[ordinal][0]\
						if isinstance(hit_fields[ordinal], (list))\
						else str(hit_fields[ordinal]))
					hit_meta["fields_found"].append(ordinal)
					ordered_hit_fields.append(my_field)
				else:
					hit_meta["fields_found"].append(ordinal)
					ordered_hit_fields.append("")
			hit_meta["results"].append(\
			"[" + str(round(score, 3)) + "] " + " ".join(ordered_hit_fields))

		self.__generate_z_score_delta(hit_meta["scores"])

		try:
			print(str(self.init_args["thread_id"]), ": ", hit_meta["results"][0])
		except IndexError:
			print("INDEX ERROR: ", transaction["DESCRIPTION"])

		return True

	@arguments(positional=["decision", "transaction", "hit_fields",\
		"z_score_delta", "business_names"])
	def __enrich_transaction(self, *args, **kwargs):
		"""Enriches the transaction if it passes the boundary"""
		enriched_transaction = kwargs["transaction"]
		hit_fields = kwargs["hit_fields"]
		field_names = self.init_args["params"]["output"]["results"]["fields"]
		fields_in_hit = [field for field in hit_fields]

		# Enrich with the fields we've found. Attach the z_score_delta
		if kwargs["decision"] == True:
			for field in field_names:
				if field in fields_in_hit:
					field_content = hit_fields[field][0] \
						if isinstance(hit_fields[field], (list)) \
						else str(hit_fields[field])
					enriched_transaction[field] = field_content
				else:
					enriched_transaction[field] = ""
			enriched_transaction["z_score_delta"] = kwargs["z_score_delta"]
		# Add a Business Name as a fall back
		if kwargs["decision"] == False:
			for field in field_names:
				enriched_transaction[field] = ""
			enriched_transaction = self.__business_name_fallback(\
				kwargs["business_names"], kwargs["transaction"])
			enriched_transaction["z_score_delta"] = 0
		return enriched_transaction

	def __generate_base_query(self, transaction, boost=1.0):
		"""Generates the basic final query used for both
		the first and second passes"""
		# Collect necessary meta info to generate query
		logger = logging.getLogger("thread " + str(self.init_args["thread_id"]))
		params = self.init_args["params"]
		result_size = self.init_args["hyperparameters"].get("es_result_size", "10")
		fields = params["output"]["results"]["fields"]
		transaction = string_cleanse(transaction["DESCRIPTION"]).rstrip()

		# If we're using masked data, remove anything with 3 X's or more
		transaction = re.sub("X{3,}", "", transaction)
		# Input transaction must not be empty
		if len(transaction) <= 2 and re.match('^[a-zA-Z0-9_]+$', transaction):
			return

		# Construct Main Query
		logger.info("BUILDING FINAL BOOLEAN SEARCH")
		bool_search = get_bool_query(size=result_size)
		bool_search["fields"] = fields
		bool_search["_source"] = "pin.*"
		should_clauses = bool_search["query"]["bool"]["should"]
		field_boosts = self.__get_boosted_fields("standard_fields")
		simple_query = get_qs_query(transaction, field_boosts, boost)
		should_clauses.append(simple_query)
		return bool_search

	def __generate_z_score_delta(self, scores):
		"""Display the Z-score delta between the first and second scores."""
		logger = logging.getLogger("thread " + str(self.init_args["thread_id"]))

		if len(scores) < 2:
			logger.info("Unable to generate Z-Score")
			return 0

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

	def __get_boosted_fields(self, vector_name):
		"""Returns a list of boosted fields built from a boost vector"""
		boost_vector = self.boost_column_vectors[vector_name]
		return [x + "^" + str(y) \
			for x, y in zip(self.boost_row_labels, boost_vector)\
			if y != 0.0]

	@arguments(positional=["thread_id", "params", "desc_queue",\
		"result_queue", "hyperparamaters"])
	def __init__(self, **kwargs):
		''' Constructor '''
		threading.Thread.__init__(self)
		self.init_args = { \
			"thread_id" : kwargs["thread_id"], \
			"params" : kwargs["params"], \
			"desc_queue" : kwargs["desc_queue"], \
			"result_queue" : kwargs["result_queue"], \
			"hyperparameters" : kwargs["hyperparamaters"] \
		}

		self.user = None
		self.cities = get_us_cities()

		cluster_nodes = self.init_args["params"]["elasticsearch"]["cluster_nodes"]
		self.es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,\
			sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)

		self.my_meta = None
		self.__reset_my_meta()
		self.__set_logger()
		self.boost_row_labels, self.boost_column_vectors = \
			self.__build_boost_vectors()

	def __load_past_transactions(self):
		"""Loads any past transactions if available"""
		# Ensure user is in index
		unique_member_id = self.user[0]["UNIQUE_MEM_ID"]
		index_body = {"user_id" : unique_member_id}
		_ = self.es_connection.index(index="user_index",\
			doc_type="user", id=unique_member_id, body=index_body)

	def __locate_user(self, unique_locations):
		"""Uses first pass results, to find an approximation
		of a users spending area. Returns estimation as
		a bounded polygon"""
		scaling_factor = self.init_args["hyperparameters"].get("scaling_factor", "1")
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
				original_geoshapes = collect_clusters(scaled_points,\
					labels, unique_locations)

			# Scale generated geo shapes
			scaled_geoshapes = [scale_polygon(geoshape,\
				scale=scaling_factor)[1] for geoshape in original_geoshapes]
			# Save interesting outputs needs to run in it's own process
			#if len(unique_locations) >= 3:
			#	pool = multiprocessing.Pool()
			#	arguments = [(unique_locations, original_geoshapes,\
			#	scaled_geoshapes, user_id)]
			#	pool.starmap(visualize, arguments)
		return scaled_geoshapes

	def __output_to_result_queue(self, enriched_transactions):
		"""Pushes results to the result queue"""
		for transaction in enriched_transactions:
			self.init_args["result_queue"].put(transaction)

	def __process_results(self, search_results, transaction):
		"""Prepare results for decision boundary"""
		field_names = self.init_args["params"]["output"]["results"]["fields"]

		# Must be at least one result
		if search_results["hits"]["total"] == 0:
			for field in field_names:
				transaction[field] = ""
				transaction["z_score_delta"] = 0
			return transaction

		# Collect Necessary Information
		hits = search_results['hits']['hits']
		scores = []
		top_hit = hits[0]
		hit_fields = top_hit.get("fields", "")
		business_names = [result.get("fields", {"name" : ""})["name"] \
			for result in hits]
		business_names = [name[0] for name in business_names \
		if isinstance(name, list)]

		# If no results return
		if hit_fields == "":
			return transaction

		# Elasticsearch v1.0 bug workaround
		if top_hit["_source"].get("pin", "") != "":
			coordinates = top_hit["_source"]["pin"]["location"]["coordinates"]
			top_hit["longitude"] = coordinates[0]
			top_hit["latitude"] = coordinates[1]

		# Collect Relevancy Scores
		for hit in hits:
			scores.append(hit['_score'])

		z_score_delta = self.__generate_z_score_delta(scores)
		decision = self.__decision_boundary(z_score_delta)

		# Enrich Data if Passes Boundary
		enriched_transaction = self.__enrich_transaction(decision,\
			transaction, hit_fields, z_score_delta, business_names)
		return enriched_transaction

	def __reset_my_meta(self):
		"""Purges several object data structures and re-initializes them."""
		self.my_meta = {"metrics": {"query_count": 0, "cache_count": 0}}

	def run(self):
		while self.init_args["desc_queue"].qsize() > 0:
			try:
				# Select all transactions from user
				self.user = self.init_args["desc_queue"].get()
				# Load Past Transactions
				self.__load_past_transactions()
				# Classify using text features only
				enriched_transactions = self.__text_features()
				# Save results to user_index
				self.__save_labeled_transactions(enriched_transactions)
				# Output Results to Result Queue
				self.__output_to_result_queue(enriched_transactions)
				# Done
				self.init_args["desc_queue"].task_done()
			except queue.Empty:
				print(str(self.init_args["thread_id"]),\
					" found empty queue, terminating.")
		return True

	def __run_classifier(self, query):
		"""Runs the classifier"""
		# Show Final Query
		logger = logging.getLogger("thread " + str(self.init_args["thread_id"]))
		logger.info(json.dumps(query, sort_keys=True, indent=4,\
		separators=(',', ': ')))
		my_results = self.__search_index(query)
		metrics = self.my_meta["metrics"]
		logger.info("Cache Hit / Miss: %i / %i", metrics["cache_count"],\
		metrics["query_count"])
		return my_results

	def __save_labeled_transactions(self, enriched_transactions):
		"""Saves the labeled transactions to our user_index"""
		for transaction in enriched_transactions:
			found_factual = transaction.get("z_score_delta", 0) > 0
			geo_available = transaction.get("longitude", "") != ""\
				and transaction.get("latitude", "") != ""
			has_date = transaction.get("TRANSACTION_DATE", "") != ""
			if found_factual and geo_available and has_date:
				self.__save_transaction(transaction)

	def __save_transaction(self, transaction):
		"""Saves a transaction to the user index"""
		transaction_id = transaction["UNIQUE_TRANSACTION_ID"]
		date = transaction["TRANSACTION_DATE"].replace(".", "-")
		date = date.replace("/", "-")
		update_body = { \
			"date": date, "_parent": transaction["UNIQUE_MEM_ID"], \
			"z_score_delta": str(transaction["z_score_delta"]), \
			"description": transaction["DESCRIPTION"], \
			"factual_id": transaction["factual_id"], \
			"pin.location": { \
				"lon" : transaction["longitude"], \
				"lat" : transaction["latitude"] \
			}
		}

		try:
			_ = self.es_connection.index(index="user_index",\
				doc_type="transaction", id=transaction_id, body=update_body,\
				routing=transaction["UNIQUE_MEM_ID"])
		except:
			logging.critical("Unable to update the following: %s",\
				str(transaction["DESCRIPTION"]))
			pprint(update_body)

	def __search_index(self, input_as_object):
		"""Searches the merchants index and the merchant mapping"""
		logger = logging.getLogger("thread " + str(self.init_args["thread_id"]))
		use_cache = self.init_args["params"]\
			["elasticsearch"].get("cache_results", True)
		input_data = json.dumps(input_as_object, sort_keys=True, indent=4\
		, separators=(',', ': ')).encode('UTF-8')
		output_data = ""

		if use_cache == True:
			output_data = self.__check_cache(input_data)

		if output_data == "":
			logger.debug("Cache miss, searching")
			try:
				output_data = self.es_connection.search(\
					index=self.init_args["params"]["elasticsearch"]["index"],\
					body=input_as_object)
				#Add newly found results to the client cache
				#if use_cache == True:
				#'input_hash' variable is undefined, this does not work!
				#self.init_args["params"]["search_cache"][input_hash] = output_data
			except:
				logging.critical("Unable to process the following: %s",\
					str(input_as_object))
				output_data = {"hits":{"total":0}}

		self.my_meta["metrics"]["query_count"] += 1
		return output_data

	def __set_logger(self):
		"""Creates a logger, based upon the supplied config object."""
		levels = {'debug': logging.DEBUG, 'info': logging.INFO,\
			'warning': logging.WARNING, 'error': logging.ERROR,\
			'critical': logging.CRITICAL}
		params = self.init_args["params"]
		my_level = params["logging"]["level"]
		if my_level in levels:
			my_level = levels[my_level]
		my_path = params["logging"]["path"]
		my_formatter = logging.Formatter(params['logging']['formatter'])
		#You'll want to add something to identify the thread
		my_logger = logging.getLogger("thread " + str(self.init_args["thread_id"]))
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

	def __text_and_geo_features(self, text_features_results):
		"""Classify transactions using geo and text features"""
		#Build 'text_features', dictionary of information about the
		#text_features_results.
		text_features = {"user_id" : text_features_results[0]['MEM_ID']}
		text_features["hits"], text_features["non_hits"] =\
			separate_geo(text_features_results)
		text_features["unique_locations"] = set([\
			str(json.loads(hit["pin.location"].replace("'", '"'))\
			["coordinates"]) for hit in text_features["hits"]])
		text_features["unique_locations"] = [\
			json.loads(location.replace("'", '"'))\
			for location in text_features["unique_locations"]]

		# Use first pass results if no location found
		scaled_geoshapes = self.__locate_user(text_features["unique_locations"])
		if not bool(scaled_geoshapes):
			return text_features_results

		#Build 'query_parts', dictionary of useful values when building
		#ES queries
		query_parts = { \
			"geo_query" : get_geo_query(scaled_geoshapes), \
			"qs_boost" : self.init_args["hyperparameters"].get("qs_boost", "1"), \
			"name_boost" : self.init_args["hyperparameters"].get("name_boost", "1") \
		}
		enriched_transactions = []

		# Run transactions again with geo_query
		for transaction in text_features["non_hits"]:
			query_parts["base_query"] = self.__generate_base_query(transaction, \
				boost=query_parts["qs_boost"])
			query_parts["should_clauses"] =\
				query_parts["base_query"]["query"]["bool"]["should"]
			query_parts["should_clauses"].append(query_parts["geo_query"])
			field_boosts = query_parts["should_clauses"][0]["query_string"]["fields"]

			for i in range(len(field_boosts)):
				if "name" in field_boosts[i]:
					field_boosts[i] = "name^" + str(query_parts["name_boost"])
				else:
					key, value = field_boosts[i].split("^")
					field_boosts[i] = key + "^" + str(float(value) * 0.5)
			search_results = self.__run_classifier(json.dumps(query_parts["base_query"]))
			self.__display_search_results(search_results, transaction)
			enriched_transaction = self.__process_results(search_results, transaction)
			enriched_transactions.append(enriched_transaction)

		added_text_and_geo_features = [trans for trans in enriched_transactions\
			if trans["factual_id"] != ""]

		text_and_geo_features_results = text_features["hits"] + enriched_transactions

		print("ADDED SECOND PASS: " + str(len(added_text_and_geo_features)))
		return text_and_geo_features_results

	def __text_features(self):
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


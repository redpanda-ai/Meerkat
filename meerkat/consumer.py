#!/usr/local/bin/python3.3

"""This module constructs ElasticSearch queries to match a transaction to
merchant data indexed with ElasticSearch. More generally it is used to
match an unstructured record to a well structured index. In our
multithreaded system, this is what processes the queue of data provided by
producer.py

Created on Jan 16, 2014
@author: J. Andrew Key
@author: Matthew Sevrens
"""

import hashlib
import json
import logging
import numpy as np
import pprint
import queue
import re
import sys
import threading
import string

from elasticsearch import Elasticsearch
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import zscore
from pprint import pprint

from .various_tools import string_cleanse, synonyms, safe_print, get_yodlee_factual_map
from .various_tools import get_bool_query, get_qs_query, get_us_cities
from .clustering import cluster, collect_clusters
from .location import separate_geo, scale_polygon, get_geo_query

PIPE_PATTERN = re.compile(r"\|")

class Consumer(threading.Thread):
	"""Acts as a client to an ElasticSearch cluster, tokenizing description
	strings that it pulls from a synchronized queue."""

	def __build_boost_vectors(self):
		"""Turns configuration entries into a dictionary of numpy arrays."""
		#logger = logging.getLogger("thread " + str(self.thread_id))
		boost_column_labels = self.hyperparameters["boost_labels"]
		boost_row_vectors = self.hyperparameters["boost_vectors"]
		boost_row_labels, boost_column_vectors = sorted(boost_row_vectors.keys()), {}
		for i in range(len(boost_column_labels)):
			my_list = []
			for field in boost_row_labels:
				my_list.append(boost_row_vectors[field][i])
			boost_column_vectors[boost_column_labels[i]] = np.array(my_list)
		return boost_row_labels, boost_column_vectors

	def __interactive_mode(self, params, scores, transaction, hits, business_names, city_names, state_names):
		"""Interact with the results as they come"""

		score = scores[0]
		z_score = self.__generate_z_score_delta(scores)
		decision = self.__decision_boundary(z_score)
		description =  ' '.join(transaction["DESCRIPTION_UNMASKED"].split())
		first_hit = hits[0]["fields"]
		fields_to_get = ["name", "region", "locality", "internal_store_number", "postcode", "address"]
		field_content = [first_hit.get(field, ["_____"])[0] for field in fields_to_get]
		city_fallback, state_fallback, name_fallback = "", "", ""

		labels = ["Transaction", "Query", "Score", "Z-Score", "Decision"]
		query = synonyms(description)
		query = string_cleanse(query)
		data = [description, query, score, z_score, decision]

		# Populate Decisions
		if not decision:
			data[4] == "No"
			fields = self.params["output"]["results"]["fields"]
			city_names = city_names[0:2]
			state_names = state_names[0:2]
			states_equal = state_names.count(state_names[0]) == len(state_names)
			city_in_transaction = (city_names[0].lower() in transaction["DESCRIPTION_UNMASKED"].lower())
			state_in_transaction = (state_names[0].lower() in transaction["DESCRIPTION_UNMASKED"].lower())
			business_names = business_names[0:2]
			top_name = business_names[0].lower()
			all_equal = business_names.count(business_names[0]) == len(business_names)
			not_a_city = top_name not in self.cities
			name_fallback = ("", business_names[0].title())[((all_equal and not_a_city)) == True]
			name_fallback = (name_fallback, transaction['GOOD_DESCRIPTION'])[(transaction['GOOD_DESCRIPTION'] != "") == True]
			name_fallback = string.capwords(name_fallback, " ")
			city_fallback = ("", city_names[0].title())[city_in_transaction == True]
			state_fallback = ("", state_names[0].upper())[(states_equal and state_in_transaction) == True]
			labels = labels + ["Name Fallback", "State Fallback", "City Fallback"]
			data = data + [name_fallback, state_fallback, city_fallback]
		else:
			data[4] = "Yes"

		# Print to User
		stats = ["{:22}: ".format(label) + "{}" for label in labels]
		stats = "\n".join(stats)
		stats = stats.format(*data)

		attr_labels = ["GOOD_DESCRIPTION"] + [field.title() for field in fields_to_get]
		attributes = ["{:22}: ".format(label) + "{}" for label in attr_labels]
		attributes = "\n".join(attributes) + "\n\n----------------------------\n"
		field_content = [transaction['GOOD_DESCRIPTION']] + field_content
		attributes = attributes.format(*field_content)
		
		prompt = stats + "\n\n" + attributes
		#print(prompt)
		user = input(prompt)

	def __generate_z_score_delta(self, scores):
		"""Generate the Z-score delta between the first and second scores."""

		logger = logging.getLogger("thread " + str(self.thread_id))

		if len(scores) < 2:
			logger.info("Unable to generate Z-Score")
			return 0

		z_scores = zscore(scores)
		first_score, second_score = z_scores[0:2]
		z_score_delta = round(first_score - second_score, 3)
		logger.info("Z-Score delta: [%.2f]", z_score_delta)

		return z_score_delta

	def __decision_boundary(self, z_score_delta):
		"""Decide whether or not we will label transaction
		by factual_id"""

		threshold = self.hyperparameters.get("z_score_threshold", "2")

		if z_score_delta is None:
			return False
		if z_score_delta > float(threshold):
			return True
		else:
			return False

	def __init__(self, thread_id, params, desc_queue, result_queue,\
		hyperparameters, cities):
		''' Constructor '''
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.desc_queue = desc_queue
		self.result_queue = result_queue
		self.user = None
		self.params = params
		self.hyperparameters = hyperparameters
		self.cities = cities

		cluster_nodes = self.params["elasticsearch"]["cluster_nodes"]
		self.es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,
			sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)

		self.my_meta = None
		self.__reset_my_meta()
		self.__set_logger()
		self.boost_row_labels, self.boost_column_vectors =\
			self.__build_boost_vectors()

	def __text_features(self):
		"""Classify transactions using text features only"""

		enriched_transactions = []

		# Call the Classifier on each transaction and return best results
		while len(self.user) > 0:
			transaction = self.user.pop()
			base_query = self.__generate_base_query(transaction)
			search_results = self.__run_classifier(base_query)
			enriched_transaction = self.__process_results(search_results, transaction)
			enriched_transactions.append(enriched_transaction)

		return enriched_transactions

	def __text_and_geo_features(self, text_features_results):
		"""Classify transactions using geo and text features"""

		user_id = text_features_results[0]['UNIQUE_MEM_ID']
		qs_boost = self.hyperparameters.get("qs_boost", "1")
		name_boost = self.hyperparameters.get("name_boost", "1")
		hits, non_hits = separate_geo(text_features_results)

		locations_found = [
		str(json.loads(hit["pin.location"].replace("'", '"'))["coordinates"])
		for hit in hits]

		unique_locations = set(locations_found)

		unique_locations = [
		json.loads(location.replace("'", '"'))
		for location in unique_locations]

		enriched_transactions = []

		# Locate user
		scaled_geoshapes = self.__locate_user(unique_locations, user_id)

		# Use first pass results if no location found
		if not bool(scaled_geoshapes):
			return text_features_results

		# Create Query
		geo_query = get_geo_query(scaled_geoshapes)

		# Run transactions again with geo_query
		for transaction in non_hits:
			base_query = self.__generate_base_query(transaction, boost=qs_boost)
			should_clauses = base_query["query"]["bool"]["should"]
			should_clauses.append(geo_query)
			field_boosts = should_clauses[0]["query_string"]["fields"]

			for i in range(len(field_boosts)):
				if "name" in field_boosts[i]:
					field_boosts[i] = "name^" + str(name_boost)
				else:
					key, value = field_boosts[i].split("^")
					field_boosts[i] = key + "^" + str(float(value) * 0.5)
			search_results = self.__run_classifier(json.dumps(base_query))
			enriched_transaction = self.__process_results(search_results, transaction)
			enriched_transactions.append(enriched_transaction)

		added_text_and_geo_features = [trans for trans in enriched_transactions
		if trans["factual_id"] != ""]

		text_and_geo_features_results = hits + enriched_transactions

		print("ADDED SECOND PASS: " + str(len(added_text_and_geo_features)))
		return text_and_geo_features_results

	def __locate_user(self, unique_locations, user_id):
		"""Uses previous results, to find an approximation
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
				original_geoshapes = collect_clusters(scaled_points,\
					labels, unique_locations)

			# Scale generated geo shapes
			scaled_geoshapes = [scale_polygon(geoshape, scale=scaling_factor)[1]
			for geoshape in original_geoshapes]

			# Save interesting outputs needs to run in it's own process
			#if len(unique_locations) >= 3:
			#	pool = multiprocessing.Pool()
			#	arguments = [(unique_locations, original_geoshapes, scaled_geoshapes, user_id)]
			#	pool.starmap(visualize, arguments)
			
		return scaled_geoshapes

	def __run_classifier(self, query):
		"""Runs the classifier"""
		# Show Final Query
		logger = logging.getLogger("thread " + str(self.thread_id))
		logger.info(json.dumps(query, sort_keys=True, indent=4, separators=(',', ': ')))
		my_results = self.__search_index(query)
		metrics = self.my_meta["metrics"]
		return my_results

	def __generate_base_query(self, transaction, boost=1.0):
		"""Generates the basic final query used for both
		the first and second passes"""
		# Collect necessary meta info to generate query
		logger = logging.getLogger("thread " + str(self.thread_id))
		params = self.params
		result_size = self.hyperparameters.get("es_result_size", "10")
		fields = params["output"]["results"]["fields"]
		good_description = transaction["GOOD_DESCRIPTION"]
		transaction = string_cleanse(transaction["DESCRIPTION_UNMASKED"]).rstrip()

		# Input transaction must not be empty
		if len(transaction) <= 2 and re.match('^[a-zA-Z0-9_]+$', transaction):
			return

		# Replace synonyms
		transaction = synonyms(transaction)
		transaction = string_cleanse(transaction)

		# Construct Main Query
		logger.info("BUILDING FINAL BOOLEAN SEARCH")
		bool_search = get_bool_query(size=result_size)
		bool_search["fields"] = fields
		bool_search["_source"] = "pin.*"
		should_clauses = bool_search["query"]["bool"]["should"]
		field_boosts = self.__get_boosted_fields("standard_fields")
		simple_query = get_qs_query(transaction, field_boosts, boost)
		should_clauses.append(simple_query)

		# Use Good Description in Query
		if good_description != "" and self.hyperparameters.get("good_description", "") != "":
			good_description_boost = self.hyperparameters["good_description"]
			name_query = get_qs_query(string_cleanse(good_description), ['name'], good_description_boost)
			should_clauses.append(name_query)

		return bool_search

	def __process_results(self, search_results, transaction):
		"""Prepare results for decision boundary"""

		field_names = self.params["output"]["results"]["fields"]
		hyperparameters = self.hyperparameters
		params = self.params

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
		
		# If no results return
		if hit_fields == "":
			return transaction

		# Collect Business Names, City Names, and State Names
		business_names = [result.get("fields", {"name" : ""}).get("name", "") for result in hits]
		business_names = [name[0] for name in business_names if type(name) == list]
		city_names = [result.get("fields", {"locality" : ""}).get("locality", "") for result in hits]
		city_names = [name[0] for name in city_names if type(name) == list]
		state_names = [result.get("fields", {"region" : ""}).get("region", "") for result in hits]
		state_names = [name[0] for name in state_names if type(name) == list]

		# Need Names
		if len(business_names) < 2:
			return transaction

		# City Names Cause issues
		if business_names[0] in self.cities:
			print("City Name: ", business_names[0], " omitted as result")
			return transaction

		# Elasticsearch v1.0 bug workaround
		if top_hit["_source"].get("pin", "") != "":
			coordinates = top_hit["_source"]["pin"]["location"]["coordinates"]
			hit_fields["longitude"] = coordinates[0]
			hit_fields["latitude"] = coordinates[1]

		# Collect Relevancy Scores
		for hit in hits:
			scores.append(hit['_score'])

		z_score_delta = self.__generate_z_score_delta(scores)
		decision = self.__decision_boundary(z_score_delta)

		# Interactive Mode (temporarily disabled)
		#if params["mode"] == "test":
			#args = [params, scores, transaction, hits, business_names, city_names, state_names]
			#self.__interactive_mode(*args)

		# Enrich Data if Passes Boundary
		args = [decision, transaction, hit_fields, z_score_delta, business_names, city_names, state_names]
		enriched_transaction = self.__enrich_transaction(*args)

		return enriched_transaction

	def __enrich_transaction(self, decision, transaction, hit_fields, z_score_delta, business_names, city_names, state_names):
		"""Enriches the transaction if it passes the boundary"""

		enriched_transaction = transaction
		field_names = self.params["output"]["results"]["fields"]
		fields_in_hit = [field for field in hit_fields]
		yfm = get_yodlee_factual_map()

		# Enrich with the fields we've found. Attach the z_score_delta
		if decision == True:
			for field in field_names:
				if field in fields_in_hit:
					field_content = hit_fields[field][0] if isinstance(hit_fields[field], (list)) else str(hit_fields[field])
					enriched_transaction[yfm.get(field, field)] = PIPE_PATTERN.sub(" ", field_content)
				else:
					enriched_transaction[yfm.get(field, field)] = ""
			enriched_transaction[yfm["z_score_delta"]] = z_score_delta

		# Add Business Name, City and State as a fallback
		if decision == False:
			for field in field_names:
				enriched_transaction[yfm.get(field, field)] = ""
			enriched_transaction = self.__business_name_fallback(business_names, transaction)
			enriched_transaction = self.__geo_fallback(city_names, state_names, transaction)
			enriched_transaction[yfm["z_score_delta"]] = 0

		# Remove Good Description
		if enriched_transaction['GOOD_DESCRIPTION'] != "":
			enriched_transaction[yfm['name']] = enriched_transaction['GOOD_DESCRIPTION']
			
		enriched_transaction["GOOD_DESCRIPTION"] = ""

		# Ensure Proper Casing
		if enriched_transaction[yfm['name']] == enriched_transaction[yfm['name']].upper():
			enriched_transaction[yfm['name']] = string.capwords(enriched_transaction[yfm['name']], " ")

		return enriched_transaction

	def __output_to_result_queue(self, enriched_transactions):
		"""Pushes results to the result queue"""

		for transaction in enriched_transactions:
			self.result_queue.put(transaction)

	def __geo_fallback(self, city_names, state_names, transaction):
		"""Basic logic to obtain a fallback for city and state
		when no factual_id is found"""

		fields = self.params["output"]["results"]["fields"]
		yfm = get_yodlee_factual_map()
		enriched_transaction = transaction
		city_names = city_names[0:2]
		state_names = state_names[0:2]
		states_equal = state_names.count(state_names[0]) == len(state_names)
		city_in_transaction = (city_names[0].lower() in enriched_transaction["DESCRIPTION_UNMASKED"].lower())
		state_in_transaction = (state_names[0].lower() in enriched_transaction["DESCRIPTION_UNMASKED"].lower())

		if (city_in_transaction):
			enriched_transaction[yfm['locality']] = city_names[0]

		if (states_equal and state_in_transaction):
			enriched_transaction[yfm['region']] = state_names[0]

		return enriched_transaction

	def __business_name_fallback(self, business_names, transaction):
		"""Basic logic to obtain a fallback for business name
		when no factual_id is found"""
		
		fields = self.params["output"]["results"]["fields"]
		yfm = get_yodlee_factual_map()

		# Default to CT Names if Available
		if transaction['GOOD_DESCRIPTION'] != "":
			transaction[yfm['name']] = transaction['GOOD_DESCRIPTION']
			return transaction

		enriched_transaction = transaction
		business_names = business_names[0:2]
		top_name = business_names[0].lower()
		all_equal = business_names.count(business_names[0]) == len(business_names)
		not_a_city = top_name not in self.cities

		if (all_equal and not_a_city):
			enriched_transaction[yfm['name']] = PIPE_PATTERN.sub(" ", business_names[0])

		return enriched_transaction

	def __reset_my_meta(self):
		"""Purges several object data structures and re-initializes them."""
		self.my_meta = {"metrics": {"query_count": 0}}

	def __search_index(self, input_as_object):
		"""Searches the merchants index and the merchant mapping"""
		logger = logging.getLogger("thread " + str(self.thread_id))
		input_data = json.dumps(input_as_object, sort_keys=True, indent=4\
		, separators=(',', ': ')).encode('UTF-8')

		output_data = ""

		try:
			output_data = self.es_connection.search(
				index=self.params["elasticsearch"]["index"], body=input_as_object)
		except Exception:
			logging.critical("Unable to process the following: %s",\
				str(input_as_object))
			output_data = {"hits":{"total":0}}

		self.my_meta["metrics"]["query_count"] += 1

		return output_data

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
		update_body = {
			"date": date,
			"_parent": transaction["UNIQUE_MEM_ID"],
			"z_score_delta": str(transaction["z_score_delta"]),
			"description": transaction["DESCRIPTION"],
			"factual_id": transaction["factual_id"],
			"pin.location": {
				"lon" : transaction["longitude"],
				"lat" : transaction["latitude"]
			}
		}

		try:
			_ = self.es_connection.index(index="user_index",\
				doc_type="transaction", id=transaction_id, body=update_body,\
				routing=transaction["UNIQUE_MEM_ID"])
		except Exception:
			logging.critical("Unable to update the following: %s",\
				str(transaction["DESCRIPTION_UNMASKED"]))
			pprint(update_body)

	def __load_past_transactions(self):
		"""Loads any past transactions if available"""
		# Ensure user is in index
		unique_member_id = self.user[0]["UNIQUE_MEM_ID"]
		index_body = {"user_id" : unique_member_id}
		_ = self.es_connection.index(index="user_index",\
			doc_type="user", id=unique_member_id, body=index_body)

	def __get_boosted_fields(self, vector_name):
		"""Returns a list of boosted fields built from a boost vector"""
		boost_vector = self.boost_column_vectors[vector_name]
		return [x + "^" + str(y)
		for x, y in zip(self.boost_row_labels, boost_vector)
		if y != 0.0]

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

				# Load Past Transactions (User Context: Currently Disabled)
				#self.__load_past_transactions()

				# Classify using text features only
				enriched_transactions = self.__text_features()

				# Save results to user_index (User Context: Currently Disabled)
				#self.__save_labeled_transactions(enriched_transactions)

				# Output Results to Result Queue
				self.__output_to_result_queue(enriched_transactions)

				# Done
				self.desc_queue.task_done()

			except queue.Empty:
				print(str(self.thread_id), " found empty queue, terminating.")

		return True

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	print("This module is a Class; it should not be run from the console.")

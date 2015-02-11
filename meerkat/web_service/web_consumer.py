#!/usr/local/bin/python3.3

"""This module enriches transactions with additional
data found by Meerkat

Created on Nov 3, 2014
@author: Matthew Sevrens
"""

import json
import string
import sys

from pprint import pprint
from scipy.stats.mstats import zscore

from meerkat.various_tools import get_es_connection, string_cleanse, get_boosted_fields
from meerkat.various_tools import synonyms, get_bool_query, get_qs_query
from meerkat.classification.load import select_model

BANK_CLASSIFIER = select_model("bank")
CARD_CLASSIFIER = select_model("card")
BANK_NPMN = select_model("bank_NPMN")
TRANSACTION_ORIGIN = select_model("transaction_type")
SUB_TRANSACTION_ORIGIN = select_model("sub_transaction_type")

class Web_Consumer():
	"""Acts as a web service client to process and enrich
	transactions in real time"""

	def __init__(self, params, hyperparams, cities):
		"""Constructor"""

		self.params = params
		self.hyperparams = hyperparams
		self.cities = cities
		self.es = get_es_connection(params)

	def __get_query(self, transaction):
		"""Create an optimized query"""

		result_size = self.hyperparams.get("es_result_size", "10")
		fields = self.params["output"]["results"]["fields"]
		transaction = string_cleanse(transaction["description"]).rstrip()

		# Input transaction must not be empty
		if len(transaction) <= 2 and re.match('^[a-zA-Z0-9_]+$', transaction):
			return

		# Replace synonyms
		transaction = synonyms(transaction)
		transaction = string_cleanse(transaction)

		# Construct Optimized Query
		o_query = get_bool_query(size=result_size)
		o_query["fields"] = fields
		should_clauses = o_query["query"]["bool"]["should"]
		field_boosts = get_boosted_fields(self.hyperparams, "standard_fields")
		simple_query = get_qs_query(transaction, field_boosts)
		should_clauses.append(simple_query)

		return o_query

	def __search_index(self, queries):
		"""Search against a structured index"""

		index = self.params["elasticsearch"]["index"]

		try:
			results = self.es.msearch(queries, index=index)
		except Exception:
			return None

		return results

	def __z_score_delta(self, scores):
		"""Find the Z-Score Delta"""

		if len(scores) < 2:
			return None

		z_scores = zscore(scores)
		first_score, second_score = z_scores[0:2]
		z_score_delta = round(first_score - second_score, 3)

		return z_score_delta

	def __process_results(self, results, transaction):
		"""Process search results and enrich transaction
		with found data"""

		params = self.params
		hyperparams = self.hyperparams
		field_names = params["output"]["results"]["fields"]

		# Must be at least one result
		if results["hits"]["total"] == 0:
			transaction = self.__no_result(transaction)
			return transaction

		# Collect Necessary Information
		hits = results['hits']['hits']
		top_hit = hits[0]
		hit_fields = top_hit.get("fields", "")
		
		# If no results return
		if hit_fields == "":
			transaction = self.__no_result(transaction)
			return transaction

		# Collect Fallback Data
		business_names = [result.get("fields", {"name" : ""}).get("name", "") for result in hits]
		business_names = [name[0] for name in business_names if type(name) == list]
		city_names = [result.get("fields", {"locality" : ""}).get("locality", "") for result in hits]
		city_names = [name[0] for name in city_names if type(name) == list]
		state_names = [result.get("fields", {"region" : ""}).get("region", "") for result in hits]
		state_names = [name[0] for name in state_names if type(name) == list]

		# Need Names
		if len(business_names) < 2:
			transaction = self.__no_result(transaction)
			return transaction

		# City Names Cause issues
		if business_names[0] in self.cities:
			transaction = self.__no_result(transaction)
			return transaction

		# Collect Relevancy Scores
		scores = [hit["_score"] for hit in hits]
		z_score_delta = self.__z_score_delta(scores)
		threshold = float(hyperparams.get("z_score_threshold", "2"))
		decision = True if (z_score_delta > threshold) else False

		# Enrich Data if Passes Boundary
		args = [decision, transaction, hit_fields, z_score_delta, business_names, city_names, state_names]
		enriched_transaction = self.__enrich_transaction(*args)

		return enriched_transaction

	def __no_result(self, transaction):
		"""Make sure transactions have proper attribute names"""

		params = self.params
		fields = params["output"]["results"]["fields"]
		labels = params["output"]["results"]["labels"]
		attr_map = dict(zip(fields, labels))

		for field in fields:
			transaction[attr_map.get(field, field)] = ""

		transaction["match_found"] = False

		return transaction

	def __enrich_transaction(self, decision, transaction, hit_fields, z_score_delta, business_names, city_names, state_names):
		"""Enriches the transaction with additional data"""

		params = self.params
		field_names = params["output"]["results"]["fields"]
		fields_in_hit = [field for field in hit_fields]
		transaction["match_found"] = False

		# Collect Mapping Details
		fields = params["output"]["results"]["fields"]
		labels = params["output"]["results"]["labels"]
		attr_map = dict(zip(fields, labels))

		# Enrich with found data
		if decision == True:
			transaction["match_found"] = True
			for field in field_names:
				if field in fields_in_hit:
					field_content = hit_fields[field][0] if isinstance(hit_fields[field], (list)) else str(hit_fields[field])
					transaction[attr_map.get(field, field)] = field_content
				else:
					transaction[attr_map.get(field, field)] = ""

		# Add Business Name, City and State as a fallback
		if decision == False:
			for field in field_names:
				transaction[attr_map.get(field, field)] = ""
			transaction = self.__business_name_fallback(business_names, transaction, attr_map)
			transaction = self.__geo_fallback(city_names, state_names, transaction, attr_map)

		# Ensure Proper Casing
		if transaction[attr_map['name']] == transaction[attr_map['name']].upper():
			transaction[attr_map['name']] = string.capwords(transaction[attr_map['name']], " ")

		# Add Source
		index = params["elasticsearch"]["index"]
		transaction["source"] = "FACTUAL" if ("factual" in index) else "OTHER"

		return transaction

	def __geo_fallback(self, city_names, state_names, transaction, attr_map):
		"""Basic logic to obtain a fallback for city and state
		when no factual_id is found"""

		fields = self.params["output"]["results"]["fields"]
		city_names = city_names[0:2]
		state_names = state_names[0:2]
		states_equal = state_names.count(state_names[0]) == len(state_names)
		city_in_transaction = (city_names[0].lower() in transaction["description"].lower())
		state_in_transaction = (state_names[0].lower() in transaction["description"].lower())

		if (city_in_transaction):
			transaction[attr_map['locality']] = city_names[0]

		if (states_equal and state_in_transaction):
			transaction[attr_map['region']] = state_names[0]

		return transaction

	def __business_name_fallback(self, business_names, transaction, attr_map):
		"""Basic logic to obtain a fallback for business name
		when no factual_id is found"""
		
		fields = self.params["output"]["results"]["fields"]
		business_names = business_names[0:2]
		top_name = business_names[0].lower()
		all_equal = business_names.count(business_names[0]) == len(business_names)
		not_a_city = top_name not in self.cities

		if (all_equal and not_a_city):
			transaction[attr_map['name']] = business_names[0]

		return transaction

	def ensure_output_schema(self, physical, non_physical):
		"""Clean output to proper schema"""

		# Add or Modify Fields
		#for trans in non_physical:
		#	trans["category"] = ""

		for trans in physical:
			categories = trans.get("category_labels", "")
			categories = json.loads(categories) if (categories != "") else []
			trans["category_labels"] = categories

		# Combine Transactions
		transactions = physical + non_physical

		# Strip Fields
		for trans in transactions:
			del trans["description"]
			del trans["amount"]
			del trans["date"]

		return transactions

	def __enrich_physical(self, transactions):
		"""Enrich physical transactions with Meerkat"""

		if len(transactions) == 0:
			return transactions

		enriched, queries = [], []
		index = self.params["elasticsearch"]["index"]

		for trans in transactions:
			query = self.__get_query(trans)
			queries.append({"index" : index})
			queries.append(query)

		queries = '\n'.join(map(json.dumps, queries))
		results = self.__search_index(queries)

		# Error Handling
		if results == None:
			return transactions

		results = results['responses']

		for r, t in zip(results, transactions):
			trans_plus = self.__process_results(r, t)
			enriched.append(trans_plus)

		return enriched

	def __enrich_non_physical(self, transactions):
		"""Enrich non-physical transactions with Meerkat"""

		if len(transactions) == 0:
			return transactions

		for trans in transactions:
			name = BANK_NPMN(trans["description"])
			trans["merchant_name"] = name.title()

		return transactions

	def __add_transaction_origin(self, data):
		"""Add transaction origin and sub origin to transaction"""

		transactions = data["transaction_list"]

		if len(transactions) == 0:
			return transactions

		for trans in transactions:
			txn_type = TRANSACTION_ORIGIN(trans["description"])
			txn_sub_type = SUB_TRANSACTION_ORIGIN(trans["description"])
			trans["txn_type"] = txn_type.title()
			trans["txn_sub_type"] = txn_sub_type.title()

		return transactions

	def __sws(self, data, transactions):
		"""Split transactions into physical and non-physical"""

		physical, non_physical = [], []

		# Determine Whether to Search
		for trans in transactions:
			classifier = BANK_CLASSIFIER if (data["container"] == "bank") else CARD_CLASSIFIER
			label = classifier(trans["description"])
			trans["is_physical_merchant"] = True if (label == "1") else False
			(non_physical, physical)[label == "1"].append(trans)

		return physical, non_physical

	def classify(self, data):
		"""Classify a set of transactions"""

		transactions = self.__add_transaction_origin(data)
		physical, non_physical = self.__sws(data, transactions)
		physical = self.__enrich_physical(physical)
		non_physical = self.__enrich_non_physical(non_physical)
		transactions = self.ensure_output_schema(physical, non_physical)
		data["transaction_list"] = transactions

		return data

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	print("This module is a Class; it should not be run from the console.")

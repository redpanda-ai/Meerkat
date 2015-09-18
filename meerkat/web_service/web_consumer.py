#!/usr/local/bin/python3.3

"""This module enriches transactions with additional
data found by Meerkat

Created on Nov 3, 2014
@author: Matthew Sevrens
"""

import json
import string
import re

from itertools import zip_longest
from pprint import pprint
from scipy.stats.mstats import zscore

from meerkat.various_tools \
import get_es_connection, string_cleanse, get_boosted_fields
from meerkat.various_tools \
import synonyms, get_bool_query, get_qs_query, load_params
from meerkat.classification.load import select_model
from meerkat.classification.lua_bridge import get_CNN
from meerkat.classification.bloom_filter.find_entities import location_split

# Enabled Models
BANK_SWS = select_model("bank")
CARD_SWS = select_model("card")
BANK_CNN = get_CNN("bank")
CARD_CNN = get_CNN("card")
BANK_SUBTYPE_CNN = get_CNN("card_subtype")
CARD_SUBTYPE_CNN = get_CNN("bank_subtype")

class Web_Consumer():
	"""Acts as a web service client to process and enrich
	transactions in real time"""

	def __init__(self, params=None, hyperparams=None, cities=None):
		"""Constructor"""
		
		if params is None:
			self.params = dict()
		else:
			self.params = params
			self.es = get_es_connection(params)

		if hyperparams is None:
			self.hyperparams = dict()
		else:
			self.hyperparams = hyperparams
		
		if cities is None:
			self.cities = dict()
		else:
			self.cities = cities

	def update_params(self, params):
		self.params = params
		self.es = get_es_connection(params)

	def update_hyperparams(self, hyperparams):
		self.hyperparams = hyperparams
	
	def update_cities(self, cities):
		self.cities = cities
	
	def __get_query(self, transaction):
		"""Create an optimized query"""

		result_size = self.hyperparams.get("es_result_size", "10")
		fields = self.params["output"]["results"]["fields"]
		locale_bloom = transaction["locale_bloom"]
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

		# Add Locale Sub Query
		if locale_bloom != None:
			city_query = get_qs_query(locale_bloom[0].lower(), ['locality'])
			state_query = get_qs_query(locale_bloom[1].lower(), ['region'])
			should_clauses.append(city_query)
			should_clauses.append(state_query)

			# add routing term
			# o_query["query"]["match"] = "%s, %s" % (locale_bloom[0], locale_bloom[1])

		return o_query

	def __search_index(self, queries):
		"""Search against a structured index"""

		index = self.params["elasticsearch"]["index"]
		# pprint(queries)

		try:
			# pull routing out of queries and append to below msearch
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

		return z_score_delta, scores[0]

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
		business_names = \
		[result.get("fields", {"name" : ""}).get("name", "") for result in hits]
		business_names = \
		[name[0] for name in business_names if type(name) == list]
		city_names = \
		[result.get("fields", {"locality" : ""}).get("locality", "") \
		for result in hits]
		city_names = \
		[name[0] for name in city_names if type(name) == list]
		state_names = \
		[result.get("fields", {"region" : ""}).get("region", "") for result in hits]
		state_names = \
		[name[0] for name in state_names if type(name) == list]

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
		z_score_delta, raw_score = self.__z_score_delta(scores)
		threshold = float(hyperparams.get("z_score_threshold", "2"))
		raw_threshold = float(hyperparams.get("raw_score_threshold", "1"))
		decision = True \
		if (z_score_delta > threshold) and (raw_score > raw_threshold) else False

		# Enrich Data if Passes Boundary
		args = [decision, transaction, hit_fields, z_score_delta, \
		business_names, city_names, state_names]
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

	def __enrich_transaction(self, decision, transaction, \
	hit_fields, z_score_delta, business_names, city_names, state_names):
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
					field_content = hit_fields[field][0] if\
 isinstance(hit_fields[field], (list)) else str(hit_fields[field])
					transaction[attr_map.get(field, field)] = field_content
				else:
					transaction[attr_map.get(field, field)] = ""

		# Add Business Name, City and State as a fallback
		if decision == False:
			for field in field_names:
				transaction[attr_map.get(field, field)] = ""
			transaction = \
			self.__business_name_fallback(business_names, transaction, attr_map)
			transaction = \
			self.__geo_fallback(city_names, state_names, transaction, attr_map)

		# Ensure Proper Casing
		if transaction[attr_map['name']] == transaction[attr_map['name']].upper():
			transaction[attr_map['name']] = \
			string.capwords(transaction[attr_map['name']], " ")

		# Add Source
		index = params["elasticsearch"]["index"]
		transaction["source"] = "FACTUAL" if ("factual" in index) and\
		 (transaction["match_found"] == True) else "YODLEE"

		return transaction

	def __geo_fallback(self, city_names, state_names, transaction, attr_map):
		"""Basic logic to obtain a fallback for city and state
		when no factual_id is found"""

		fields = self.params["output"]["results"]["fields"]
		city_names = city_names[0:2]
		state_names = state_names[0:2]
		states_equal = \
		state_names.count(state_names[0]) == len(state_names)
		city_in_transaction = \
		(city_names[0].lower() in transaction["description"].lower())
		state_in_transaction = \
		(state_names[0].lower() in transaction["description"].lower())

		if city_in_transaction:
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

	def ensure_output_schema(self, data, physical, non_physical):
		"""Clean output to proper schema"""

		# Add or Modify Fields
		for trans in physical:
			categories = trans.get("category_labels", "")
			categories = json.loads(categories) if (categories != "" and categories != []) else []
			trans["category_labels"] = categories

		# Combine Transactions
		transactions = physical + non_physical

		# Collect Mapping Details
		fields = self.params["output"]["results"]["fields"]
		labels = self.params["output"]["results"]["labels"]
		attr_map = dict(zip(fields, labels))

		# Override / Strip Fields
		for trans in transactions:

			# Override output with CNN v1
			if trans["CNN"] != "":
				trans[attr_map["name"]] = trans["CNN"]

			# Override Locale with Bloom Results
			if trans["locale_bloom"] != None and trans["is_physical_merchant"] == True:
				trans["city"] = trans["locale_bloom"][0]
				trans["state"] = trans["locale_bloom"][1]

			del trans["locale_bloom"]
			del trans["description"]
			del trans["amount"]
			del trans["date"]
			del trans["CNN"]
			del trans["ledger_entry"]

		return transactions

	def __enrich_physical(self, transactions):
		"""Enrich physical transactions with Meerkat"""

		if len(transactions) == 0:
			return transactions

		enriched, queries = [], []
		index = self.params["elasticsearch"]["index"]

		for trans in transactions:
			query = self.__get_query(trans)

			# add routing to header
			#try:
			#	locality = query['query']['bool']['should'][1]['query_string']['query']
			#	region = query['query']['bool']['should'][2]['query_string']['query']
			queries.append({"index" : index})#, "routing" : "%s%s" % (locality, region)})
			#except IndexError:
			#	queries.append({"index" : index})
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

	def __sws(self, data, transactions):
		"""Split transactions into physical and non-physical"""

		physical, non_physical = [], []

		# Determine Whether to Search
		for trans in transactions:
			classifier = BANK_SWS if (data["container"] == "bank") else CARD_SWS
			label = classifier(trans["description"])
			trans["is_physical_merchant"] = True if (label == "1") else False
			(non_physical, physical)[label == "1"].append(trans)

		return physical, non_physical

	def __apply_merchant_CNN(self, data, transactions):
		"""Apply the merchant CNN to transactions"""

		classifier = BANK_CNN if (data["container"] == "bank") else CARD_CNN

		processed = classifier(transactions)

		return processed

	def __apply_subtype_CNN(self, data):
		"""Apply the subtype CNN to transactions"""

		transactions = data["transaction_list"]
		classifier = BANK_SUBTYPE_CNN if (data["container"] == "bank") else CARD_SUBTYPE_CNN

		if len(transactions) == 0:
			return transactions

		processed = classifier(transactions, label_key="subtype_CNN")

		for t in processed:
			txn_type, txn_sub_type = t["subtype_CNN"].split(" - ")
			t["txn_type"] = txn_type
			t["txn_sub_type"] = txn_sub_type
			del t["subtype_CNN"]

		return processed

	def __apply_locale_bloom(self, data, transactions):
		""" Apply the locale bloom filter to transactions"""

		for trans in transactions:
			try:
				description = trans["description"]
				trans["locale_bloom"] = location_split(description)
			except KeyError:
				pass

		return transactions

	def classify(self, data):
		"""Classify a set of transactions"""

		transactions = self.__apply_subtype_CNN(data)
		transactions = self.__apply_locale_bloom(data, transactions)
		transactions = self.__apply_merchant_CNN(data, transactions)
		physical, non_physical = self.__sws(data, transactions)
		physical = self.__enrich_physical(physical)
		transactions = self.ensure_output_schema(data, physical, non_physical)
		data["transaction_list"] = transactions

		return data

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	print("This module is a Class; it should not be run from the console.")

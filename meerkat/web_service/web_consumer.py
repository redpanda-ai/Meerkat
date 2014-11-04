#!/usr/local/bin/python3.3

"""This module enriches transactions with additional
data found by Meerkat

Created on Nov 3, 2014
@author: Matthew Sevrens
"""

import json

from pprint import pprint

from meerkat.various_tools import get_es_connection, string_cleanse, get_boosted_fields
from meerkat.various_tools import synonyms, get_bool_query, get_qs_query
from meerkat.binary_classifier.load import select_model

BANK_CLASSIFIER = select_model("bank")
CARD_CLASSIFIER = select_model("card")

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
		o_query["_source"] = "*"
		should_clauses = o_query["query"]["bool"]["should"]
		field_boosts = get_boosted_fields(self.hyperparams, "standard_fields")
		simple_query = get_qs_query(transaction, field_boosts)
		should_clauses.append(simple_query)

		return o_query

	def __enrich_physical(self, transactions):
		"""Enrich physical transactions with Meerkat"""

		enriched = []

		for trans in transactions:
			query =  self.__get_query(trans)
			pprint(query)
			# Generate Query
			# Search Index
			# Add Results

		enriched = transactions

		return enriched

	def __sws(self, transactions):
		"""Split transactions into physical and non-physical"""

		physical, non_physical = [], []

		# Determine Whether to Search
		for trans in transactions:
			label = BANK_CLASSIFIER(trans["description"])
			trans["is_physical_merchant"] = True if (label == "1") else False
			(non_physical, physical)[label == "1"].append(trans)

		return physical, non_physical

	def classify(self, data):
		"""Classify a set of transactions"""

		physical, non_physical = self.__sws(data["transaction_list"])
		physical = self.__enrich_physical(physical)
		transactions = physical + non_physical
		data["transaction_list"] = transactions

		#print(json.dumps(data))

		return {}

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	print("This module is a Class; it should not be run from the console.")
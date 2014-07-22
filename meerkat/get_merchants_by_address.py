#!/usr/local/bin/python3.3

"""This script takes a list of transactions
and their matched addresses and returns a
document with associated factual_id's. 

Created on July 21, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: In Progress
# python3.3 -m meerkat.get_merchants_by_address config/test.json data/misc/ground_truth_card.txt

# Required Columns: 
# PHYSICAL_MERCHANT: Name found in factual
# CITY: City name to match against
# STREET: Address to match against
# STATE: State to match against

#####################################################

import os
import sys

from pprint import pprint
from scipy.stats.mstats import zscore

from meerkat.description_consumer import get_qs_query, get_bool_query
from meerkat.various_tools import load_params, get_es_connection, string_cleanse
from meerkat.various_tools import get_merchant_by_id, load_dict_list, write_dict_list

def enrich_transactions(params, es_connection):
	"""Enrich a set of transactions using a provided factual_id"""

	transactions = load_dict_list(sys.argv[2])

	for transaction in transactions:

		merchant = find_merchant_by_address(params, transaction, es_connection)

def find_merchant_by_address(params, store, es_connection):
	"""Match document with address to factual document"""

	fields = ["name", "address", "locality", "region"]
	search_parts = [store["PHYSICAL_MERCHANT"], store["STREET"], store["CITY"], string_cleanse(store["STATE"])]
	transaction = store["DESCRIPTION_UNMASKED"]
	index = params["elasticsearch"]["index"]
	factual_id = ""
	result = ""

	# Generate Query
	bool_search = get_bool_query(size=45)
	should_clauses = bool_search["query"]["bool"]["should"]

	# Multi Field
	for i in range(len(fields)):
		sub_query = get_qs_query(search_parts[i], [fields[i]])
		should_clauses.append(sub_query)

	# Search
	try:
		results = es_connection.search(index=index, body=bool_search)
	except Exception:
		results = {"hits":{"total":0}}

	score, top_hit = get_hit(results, 0)

	if score == False:
		return "", ""

	# Tests
	if top_hit.get("factual_id", "") != store["factual_id"]:

		# Print results
		fields_to_print = ["name", "address", "locality", "region", "postcode", "internal_store_number"]
		formatted = [top_hit.get(field, "") for field in fields_to_print]
		formatted = ", ".join(formatted)
		print("DESCRIPTION_UNMASKED: ", transaction)
		print("Query Sent: ", search_parts, " ")
		print("Top Result: ", formatted.encode("utf-8"))
		print("Z-Score: ", score, "\n")
		print("Old ID ", store["factual_id"])
		print("New ID ", top_hit["factual_id"], "\n")

		input_var = input("Enter something: ")

def get_hit(search_results, index):

	# Must have results
	if search_results['hits']['total'] == 0:
		return False, False

	hits = search_results['hits']['hits']
	scores = [hit['_score'] for hit in hits]
	z_score = z_score_delta(scores)
	hit = hits[index]

	return z_score, hit["_source"]

def z_score_delta(scores):
	"""Find the Z-Score Delta"""

	if len(scores) < 2:
		return None

	z_scores = zscore(scores)
	first_score, second_score = z_scores[0:2]
	z_score_delta = round(first_score - second_score, 3)

	return z_score_delta

def verify_arguments():
	"""Verify Usage"""

	sufficient_arguments = (len(sys.argv) == 3)

	if not sufficient_arguments:
		print("Insufficient arguments. Please see usage")
		sys.exit()

	config = sys.argv[1]
	factual_list = sys.argv[2]

	config_included = config.endswith('.json')
	factual_list_included = factual_list.endswith('.txt')

	if not config_included  or not factual_list_included:
		print("Erroneous arguments. Please see usage")
		sys.exit()

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""

	verify_arguments()
	params = load_params(sys.argv[1])
	es_connection = get_es_connection(params)
	enrich_transactions(params, es_connection)
	
if __name__ == "__main__":
	run_from_command_line(sys.argv)
#!/usr/local/bin/python3.3

"""This script takes a list of transactions
and their matched addresses and returns a
document with associated factual_id's. 

Created on July 21, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: In Progress
# python3.3 -m meerkat.compare_indices [config_file] [labeled_transactions] [old_index] [new_index]
# python3.3 -m meerkat.compare_indices config/test.json data/misc/ground_truth_card.txt factual_index factual_index_2

# Required Columns: 
# DESCRIPTION_UNMASKED
# UNIQUE_MEM_ID
# factual_id

#####################################################

import os
import sys

from pprint import pprint
from copy import deepcopy
from scipy.stats.mstats import zscore

from meerkat.description_consumer import get_qs_query, get_bool_query
from meerkat.various_tools import load_params, get_es_connection, string_cleanse
from meerkat.various_tools import get_merchant_by_id, load_dict_list, write_dict_list

def relink_transactions(params, es_connection):
	"""Relink transactions to their new factual_ids"""

	transactions = load_dict_list(sys.argv[2])

	for transaction in transactions:

		# Null
		if transaction["factual_id"] == "NULL":
			params["compare_indices"]["NULL"].append(transaction)
			continue

		# Compare Indices
		old_mapping = enrich_transaction(params, transaction, es_connection, index=sys.argv[3])
		new_mapping = enrich_transaction(params, transaction, es_connection, index=sys.argv[4])

		if new_mapping["merchant_found"] == False:
			params["compare_indices"]["id_changed"].append(transaction)
			continue

		mapping_changed = has_mapping_changed(params, old_mapping, new_mapping)

		# Reconcile Mapping
		#if mapping_changed:
			#merchant = find_merchant_by_address(params, old_mapping, es_connection)

	print("Percent Changed Factual: ", len(params["compare_indices"]["id_changed"]) / len(transactions))
	print("Percent Changed Details: ", len(params["compare_indices"]["details_changed"]) / len(transactions))

def has_mapping_changed(params, old_mapping, new_mapping):
	"""Compares two transactions for similarity"""

	fields_to_compare = ["PHYSICAL_MERCHANT", "STREET", "CITY", "STATE", "ZIP_CODE"]
	old_fields = [old_mapping.get(field, "") for field in fields_to_compare]
	new_fields = [new_mapping.get(field, "") for field in fields_to_compare]

	if old_fields != new_fields:
		params["compare_indices"]["details_changed"].append(new_mapping)
		return True

	return False

def enrich_transaction(params, transaction, es_connection, index=""):
	"""Enrich a set of transactions using a provided factual_id"""

	transaction = deepcopy(transaction)
	transaction["merchant_found"] = True
	fields_to_get = ["name", "region", "locality", "internal_store_number", "postcode", "address"]
	merchant = get_merchant_by_id(params, transaction["factual_id"], es_connection, index=index, fields=fields_to_get)

	if merchant == None:
		merchant = {}
		transaction["merchant_found"] = False
 
	transaction["PHYSICAL_MERCHANT"] = merchant.get("name", "")
	transaction["STORE_NUMBER"] = merchant.get("internal_store_number", "")
	transaction["STREET"] = merchant.get("address", "")
	transaction["CITY"] = merchant.get("locality", "")
	transaction["STATE"] = merchant.get("region", "")
	transaction["ZIP_CODE"] = merchant.get("postcode", "")

	return transaction

def find_merchant_by_address(params, store, es_connection):
	"""Match document with address to factual document"""

	fields = ["name", "address", "locality", "region", "postcode"]
	search_parts = [store["PHYSICAL_MERCHANT"], store["STREET"], store["CITY"], string_cleanse(store["STATE"]), store["ZIP_CODE"]]
	transaction = store["DESCRIPTION_UNMASKED"]
	index = sys.argv[4]
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
		formatted_new = [top_hit.get(field, "") for field in fields_to_print]
		formatted_new = ", ".join(formatted_new)
		search_parts = [part.encode("utf-8") for part in search_parts]

		print("DESCRIPTION_UNMASKED: ", transaction)
		print("Query Sent: ", search_parts, " ")
		print("Top Result: ", formatted_new.encode("utf-8"))
		print("Z-Score: ", score, "\n")
		print("Old ID ", store["factual_id"])
		print("New ID ", top_hit["factual_id"], "\n")

		#input_var = input("Enter something: ")

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

	sufficient_arguments = (len(sys.argv) == 5)

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

def add_local_params(params):
	"""Adds additional local params"""

	params["compare_indices"] = {}
	params["compare_indices"]["NULL"] = []
	params["compare_indices"]["id_changed"] = []
	params["compare_indices"]["details_changed"] = []

	return params

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""

	verify_arguments()
	params = load_params(sys.argv[1])
	params = add_local_params(params)
	es_connection = get_es_connection(params)
	relink_transactions(params, es_connection)
	
if __name__ == "__main__":
	run_from_command_line(sys.argv)
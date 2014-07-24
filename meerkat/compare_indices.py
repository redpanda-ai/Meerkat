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
import random
import contextlib

from pprint import pprint
from copy import deepcopy
from scipy.stats.mstats import zscore

from meerkat.description_consumer import get_qs_query, get_bool_query
from meerkat.various_tools import load_params, get_es_connection, string_cleanse
from meerkat.various_tools import get_merchant_by_id, load_dict_list, write_dict_list

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostderr():
    save_stderr = sys.stderr
    sys.stderr = DummyFile()
    yield
    sys.stderr = save_stderr

def relink_transactions(params, es_connection):
	"""Relink transactions to their new factual_ids"""

	# Locate Changes
	identify_changes(params, es_connection)

	# Fix Changes
	reconcile_changed_ids(params, es_connection)
	reconcile_changed_details(params, es_connection)
	reconcile_null(params, es_connection)

def identify_changes(params, es_connection):
	"""Locate changes in training data between indices"""

	transactions = load_dict_list(sys.argv[2])

	print("Locating changes in training set:")

	for i, transaction in enumerate(transactions):

		# Progress
		progress = (i / len(transactions)) * 100
		progress = str(round(progress, 0)) + "%"
		sys.stdout.write('\r')
		sys.stdout.write(progress)
		sys.stdout.flush()

		# Null
		if transaction["factual_id"] == "NULL":
			params["compare_indices"]["NULL"].append(transaction)
			continue

		# Compare Indices
		old_mapping = enrich_transaction(params, transaction, es_connection, index=sys.argv[3])
		new_mapping = enrich_transaction(params, transaction, es_connection, index=sys.argv[4])

		if new_mapping["merchant_found"] == False:
			params["compare_indices"]["id_changed"].append(old_mapping)
			continue

		mapping_changed = has_mapping_changed(params, old_mapping, new_mapping)

		if mapping_changed:
			params["compare_indices"]["details_changed"].append(transaction)
			continue

		params["compare_indices"]["no_change"].append(transaction)

	print_diff_stats(params, transactions)

def print_diff_stats(params, transactions):
	"""Display a set of diff stats"""

	sys.stdout.write('\n\n')
	print("Number of transactions: ", len(transactions))
	print("Number Changed ID: ", len(params["compare_indices"]["id_changed"]))
	print("Number Changed Details: ", len(params["compare_indices"]["details_changed"]))
	print("Number Unchanged: ", len(params["compare_indices"]["no_change"]))
	print("Number Null: ", len(params["compare_indices"]["NULL"]), "\n")

def has_mapping_changed(params, old_mapping, new_mapping):
	"""Compares two transactions for similarity"""

	fields_to_compare = ["PHYSICAL_MERCHANT", "STREET", "CITY", "STATE", "ZIP_CODE"]
	old_fields = [old_mapping.get(field, "") for field in fields_to_compare]
	new_fields = [new_mapping.get(field, "") for field in fields_to_compare]

	if old_fields != new_fields:
		return True

	return False

def reconcile_changed_ids(params, es_connection):
	"""Attempt to find a matching factual_id using address"""

	while len(params["compare_indices"]["id_changed"]) > 0:
		print("------------------", "\n")
		changed_ids = params["compare_indices"]["id_changed"]
		random.shuffle(changed_ids)
		transaction = changed_ids.pop()
		results = find_merchant_by_address(params, transaction, es_connection)
		decision_boundary(params, transaction, results)

def reconcile_changed_details(params, es_connection):
	"""Decide whether new details are erroneous"""

def reconcile_null(params, es_connection):
	"""Attempt to find a factual_id for a NULL entry"""

	null = params["compare_indices"]["NULL"]
	null_len = len(null)
	skipped = params["compare_indices"]["skipped"]
	params["compare_indices"]["NULL"] = []
	params["compare_indices"]["skipped"] = []

	# Do null then skipped
	while len(null) > 0:
		print("------------------", "\n")
		progress = 100 - ((len(null) / null_len) * 100)
		print(round(progress, 2), "% ", "done with NULL")
		transaction = null.pop()
		results = search_with_user_input(params, es_connection, transaction)
		if results == False:
			continue
		null_decision_boundary(params, transaction, results)

def search_with_user_input(params, es_connection, transaction):
	"""Search for a location by providing additional data"""

	index = sys.argv[4]
	prompt = "Base query: " + transaction["DESCRIPTION_UNMASKED"]
	prompt = prompt + " is insufficient to uniquely identify, please provide additional data \n"
	user_input = input(prompt)

	if user_input == "rm":
		return False

	# Generate new query
	bool_search = get_bool_query(size=45)
	should_clauses = bool_search["query"]["bool"]["should"]

	trans_query = get_qs_query(transaction["DESCRIPTION_UNMASKED"], ["_all"])
	user_query = get_qs_query(user_input, ["_all"])
	should_clauses.append(trans_query)
	should_clauses.append(user_query)

	# Search
	try:
		results = es_connection.search(index=index, body=bool_search)
	except Exception:
		results = {"hits":{"total":0}}

	return results

def null_decision_boundary(params, store, results):
	"""Decide if there is a match when evaluating NULL"""

	accepted_inputs = [str(x) for x in list(range(5))]

	for i in range(5):
		print_formatted_result(results, i)

	print("[enter] NULL")
	print("[rm] transaction is not physical, remove it from data", "\n")

	user_input = input("Please select a choice above: \n")

	# Remove from Set
	if user_input == "rm":
		return

	# Change factual_id, move to relinked
	if user_input in accepted_inputs:
		score, hit = get_hit(results, int(user_input))
		store["new_id"] = hit["factual_id"]
		params["compare_indices"]["relinked"].append(store)
	else:
		# Add transaction to another queue for later analysis
		params["compare_indices"]["NULL"].append(store)

def decision_boundary(params, store, results):
	"""Decide if there is a match"""

	fields = ["name", "address", "locality", "region", "postcode"]
	old_details = [store["PHYSICAL_MERCHANT"], store["STREET"], store["CITY"], string_cleanse(store["STATE"]), store["ZIP_CODE"]]
	accepted_inputs = [str(x) for x in list(range(5))]
	score, top_hit = get_hit(results, 0)

	# Add transaction back to the queue for later analysis if nothing found
	if score == False:
		print("No matches found", "\n")
		params["compare_indices"]["skipped"].append(store)
		return

	# Compare Results
	old_details = [part.encode("utf-8") for part in old_details]
	new_details = [top_hit.get(field, "").encode("utf-8") for field in fields[0:5]]

	# Don't require input on matching details
	if new_details == old_details:
		print("Record autolinked", "\n")
		user_input = "0"
	else:
		user_input = user_prompt(params, old_details, results, store)

	# Remove is transaction isn't physical
	if user_input == "rm":
		return

	# Change factual_id, move to relinked
	if user_input in accepted_inputs:
		score, hit = get_hit(results, int(user_input))
		store["new_id"] = hit["factual_id"]
		params["compare_indices"]["relinked"].append(store)
	else:
		# Add transaction to another queue for later analysis
		params["compare_indices"]["skipped"].append(store)

def enrich_transaction(params, transaction, es_connection, index=""):
	"""Return a copy of a transaction, enriched with data from a 
	provided factual_id"""

	transaction = deepcopy(transaction)
	transaction["merchant_found"] = True
	fields_to_get = ["name", "region", "locality", "internal_store_number", "postcode", "address"]
	
	# Get merchant and suppress errors
	with nostderr():
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

def find_merchant_by_address(params, store, es_connection, additional_data=""):
	"""Match document with address to factual document"""

	fields = ["name^2", "address", "locality", "region", "postcode"]
	old_details = [store["PHYSICAL_MERCHANT"], store["STREET"], store["CITY"], string_cleanse(store["STATE"]), store["ZIP_CODE"]]
	index = sys.argv[4]
	results = ""

	# Append additional data
	fields.append("_all")
	old_details.append(additional_data)

	# Generate Query
	bool_search = get_bool_query(size=45)
	should_clauses = bool_search["query"]["bool"]["should"]

	# Multi Field
	for i in range(len(fields)):
		sub_query = get_qs_query(old_details[i], [fields[i]])
		should_clauses.append(sub_query)

	# Search
	try:
		results = es_connection.search(index=index, body=bool_search)
	except Exception:
		results = {"hits":{"total":0}}

	return results

def user_prompt(params, old_details, results, store):
	"""Prompt a user for input to continue"""

	num_changed = len(params["compare_indices"]["id_changed"])
	num_relinked = len(params["compare_indices"]["relinked"])
	num_skipped = len(params["compare_indices"]["skipped"])
	total = num_changed + num_relinked + num_skipped
	percentage_relinked = num_relinked / total
	percentage_formatted = str(round(percentage_relinked * 100, 2)) + "%"
	old_details_formatted = [detail.decode("utf-8") for detail in old_details]
	old_details_formatted =  ", ".join(old_details_formatted)

	print("Number id_changed remaining: ", num_changed)
	print("Number relinked: ", num_relinked)
	print("Number skipped: ", num_skipped)
	print("Percent Relinked: ", percentage_formatted, "\n")

	print("DESCRIPTION_UNMASKED: ", store["DESCRIPTION_UNMASKED"].encode("utf-8"))
	print("Query Sent: ", old_details_formatted.encode("utf-8"), " ")
	
	for i in range(5):
		print_formatted_result(results, i)

	print("[enter] None of the above")
	print("[rm] transaction is not physical, remove it from data", "\n")
	user_input = input("Please select a location, or press enter to skip: \n")

	return user_input

def print_formatted_result(results, index):
	"""Display a potential result in readable format"""

	fields_to_print = ["name", "address", "locality", "region", "postcode", "internal_store_number"]
	score, hit = get_hit(results, index)
	details_formatted = [hit.get(field, "") for field in fields_to_print]
	details_formatted = ", ".join(details_formatted)
	print("[" + str(index) + "]", details_formatted.encode("utf-8"))

def validate_hit():
	"""Rely on user input to validate a potential match"""

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
	params["compare_indices"]["skipped"] = []
	params["compare_indices"]["no_change"] = []
	params["compare_indices"]["relinked"] = []

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
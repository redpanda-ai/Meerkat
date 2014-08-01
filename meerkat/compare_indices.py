#!/usr/local/bin/python3.3

"""This module takes a file containing
transactions and their associated
uuid relative to a specific index, 
and helps to reconcile changes as the 
index evolves over time.

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
# GOOD_DESCRIPTION

#####################################################

import os
import sys
import random
import contextlib
import collections

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

	# Generate User Context
	generate_user_context(params, es_connection)

	# Locate Changes
	identify_changes(params, es_connection)

	# Select Mode and Fix Changes
	mode_change(params, es_connection)
	
	# Save Changes
	save_relinked_transactions(params)

def mode_change(params, es_connection):
	"""Define what mode to work in"""

	mode = ""
	accepted_inputs = [str(x) for x in list(range(4))]

	safe_print("Which task would you like to complete? \n")
	safe_print("[0] Run all tasks")
	safe_print("[1] Relink transactions where factual_id no longer exists")
	safe_print("[2] Verify changes to merchants")
	safe_print("[3] Link transactions with NULL factual_id \n")

	while mode not in accepted_inputs: 
		mode = safe_input()
	
	if mode == "0":
		run_all_modes(params, es_connection)
	elif mode == "1": 
		reconcile_changed_ids(params, es_connection)
	elif mode == "2":
		reconcile_changed_details(params, es_connection)
		reconcile_changed_ids(params, es_connection)
	elif mode == "3":
		reconcile_null(params, es_connection)

def run_all_modes(params, es_connection):
	"""Run the entire relinking program"""

	reconcile_changed_details(params, es_connection)
	reconcile_changed_ids(params, es_connection)
	reconcile_null(params, es_connection)

	# Save Null
	null = params["compare_indices"]["NULL"]

	for n in null:
		n["relinked_id"] = "NULL"

def generate_user_context(params, es_connection):
	"""Generate a list of cities common to a user"""

	transactions = load_dict_list(sys.argv[2])

	sys.stdout.write('\n')
	print("Generating User Context:")

	for i, transaction in enumerate(transactions):

		# Progress
		progress = (i / len(transactions)) * 100
		progress = str(round(progress, 0)) + "%"
		sys.stdout.write('\r')
		sys.stdout.write(progress)
		sys.stdout.flush()

		# Save Context
		enriched = enrich_transaction(params, transaction, es_connection, index=sys.argv[3])
		unique_mem_id = enriched["UNIQUE_MEM_ID"]
		location = enriched["CITY"] + ", " + enriched["STATE"]

		if location not in params["compare_indices"]["user_context"][unique_mem_id] and location != ", ":
			params["compare_indices"]["user_context"][unique_mem_id].append(location)

	sys.stdout.write('\n\n')

def identify_changes(params, es_connection):
	"""Locate changes in training data between indices"""

	transactions = load_dict_list(sys.argv[2])

	safe_print("Locating changes in training set:")

	for i, transaction in enumerate(transactions):

		# Progress
		progress = (i / len(transactions)) * 100
		progress = str(round(progress, 0)) + "%"
		sys.stdout.write('\r')
		sys.stdout.write(progress)
		sys.stdout.flush()

		# Add a field for tracking
		if transaction.get("relinked_id", "") == "":
			transaction["relinked_id"] = ""
		else:
			params["compare_indices"]["relinked"].append(transaction)
			continue

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

		transaction["relinked_id"] = transaction["factual_id"]
		params["compare_indices"]["relinked"].append(transaction)

	print_diff_stats(params, transactions)

def print_diff_stats(params, transactions):
	"""Display a set of diff stats"""

	sys.stdout.write('\n\n')
	safe_print("{:25}{}".format("Number of transactions: ", len(transactions)))
	safe_print("{:25}{}".format("No action necessary: ", len(params["compare_indices"]["relinked"])))
	safe_print("{:25}{}".format("Number Changed ID: ", len(params["compare_indices"]["id_changed"])))
	safe_print("{:25}{}".format("Number Changed Details: ", len(params["compare_indices"]["details_changed"])))
	safe_print("{:25}{}\n".format("Number NULL: ", len(params["compare_indices"]["NULL"])))

def print_current_stats(params):
	"""Print Current Stats"""

	sys.stdout.write('\n')
	safe_print("{:25}{}".format("No action necessary: ", len(params["compare_indices"]["relinked"])))
	safe_print("{:25}{}".format("Number Changed ID: ", len(params["compare_indices"]["id_changed"])))
	safe_print("{:25}{}".format("Number Changed Details: ", len(params["compare_indices"]["details_changed"])))
	safe_print("{:25}{}\n".format("Number NULL: ", len(params["compare_indices"]["NULL"])))

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

	prompt_mode_change("changed id")

	while len(params["compare_indices"]["id_changed"]) > 0:
		br()
		changed_ids = params["compare_indices"]["id_changed"]
		random.shuffle(changed_ids)
		transaction = changed_ids.pop()
		results = find_merchant_by_address(params, transaction, es_connection)
		decision_boundary(params, transaction, results)

	print_current_stats(params)

	reconcile_skipped(params, es_connection)

def reconcile_changed_details(params, es_connection):
	"""Decide whether new details are erroneous"""

	prompt_mode_change("changed details")
	total = len(params["compare_indices"]["details_changed"])

	while len(params["compare_indices"]["details_changed"]) > 0:

		# Prepare Data
		br()
		details_changed = params["compare_indices"]["details_changed"]
		random.shuffle(details_changed)
		transaction = details_changed.pop()
		old_mapping = enrich_transaction(params, transaction, es_connection, index=sys.argv[3])
		new_mapping = enrich_transaction(params, transaction, es_connection, index=sys.argv[4])

		# Track Task Completion
		percent_done = (1 - (len(details_changed) / total)) * 100
		safe_print(str(round(percent_done, 1)) + "% " + "done with details changed mode \n")

		# Inform User
		safe_print("DESCRIPTION_UNMASKED: ", transaction["DESCRIPTION_UNMASKED"])

		old_details = [old_mapping["PHYSICAL_MERCHANT"], old_mapping["STREET"], old_mapping["CITY"], string_cleanse(old_mapping["STATE"]), old_mapping["ZIP_CODE"], old_mapping["STORE_NUMBER"]]
		old_details = [detail if detail != "" else "_____" for detail in old_details]
		old_details_formatted = ", ".join(old_details).encode("utf-8", "replace")
		safe_print("Old index details: " + str(old_details_formatted))

		new_details = [new_mapping["PHYSICAL_MERCHANT"], new_mapping["STREET"], new_mapping["CITY"], string_cleanse(new_mapping["STATE"]), new_mapping["ZIP_CODE"], new_mapping["STORE_NUMBER"]]
		new_details = [detail if detail != "" else "_____" for detail in new_details]
		new_details_formatted = ", ".join(new_details).encode("utf-8", "replace")
		safe_print("New index details: " + str(new_details_formatted))
		sys.stdout.write('\n')

		# Prompt User
		safe_print("Is the new merchant correct?")
		safe_print("[enter] Yes")
		safe_print("{:7s} Not Sure".format("[n]"))

		choice = safe_input()

		# Take Action
		if choice == "n":
			params["compare_indices"]["id_changed"].append(transaction)
		else:
			transaction["relinked_id"] = transaction["factual_id"]
			params["compare_indices"]["relinked"].append(transaction)

def prompt_mode_change(mode):
	"""Prompt a user that the mode has changed"""

	break_point = ""
	while break_point != "OK":
		break_point = safe_input("--- Entering " + mode + " mode. Type OK to continue --- \n")
		if break_point == "EXIT":
			save_relinked_transactions(params)
			sys.exit()

def reconcile_skipped(params, es_connection):
	"""Attempt to reconcile transactions skipped
	during changed id mode"""

	skipped = params["compare_indices"]["skipped"]
	params["compare_indices"]["skipped"] = []
	skipped_len = len(skipped)

	# Prompt a mode change
	prompt_mode_change("skipped")

	# Fix Skipped
	while len(skipped) > 0:
		br()
		progress = 100 - ((len(skipped) / skipped_len) * 100)
		safe_print(round(progress, 2), "% ", "done with Skipped")
		transaction = skipped.pop()
		name, address = skipped_details_prompt(params, transaction, es_connection)
		extra_queries = [(["address", "locality", "region", "postcode"], address, 3), (["name"], name, 4)]
		results = find_merchant_by_address(params, transaction, es_connection, additional_data=extra_queries)
		if results == False:
			continue
		null_decision_boundary(params, transaction, results)

	print_current_stats(params)

def reconcile_null(params, es_connection):
	"""Attempt to find a factual_id for a NULL entry"""

	null = params["compare_indices"]["NULL"]
	params["compare_indices"]["NULL"] = []
	null_len = len(null)
	random.shuffle(null)

	# Prompt a mode change
	prompt_mode_change("NULL")

	# Fix Null
	while len(null) > 0:
		br()
		progress = 100 - ((len(null) / null_len) * 100)
		safe_print(round(progress, 2), "% ", "done with NULL")
		transaction = null.pop()
		results = search_with_user_safe_input(params, es_connection, transaction)
		if results == False:
			continue
		null_decision_boundary(params, transaction, results)

	print_current_stats(params)

def save_relinked_transactions(params):
	""""Save the completed file set"""

	safe_print("What should the output file be named? \n")
	file_name = ""

	while file_name == "":
		file_name = safe_input()

	changed_details = params["compare_indices"]["details_changed"]
	null = params["compare_indices"]["NULL"]
	relinked = params["compare_indices"]["relinked"]

	transactions = changed_details + null + relinked
	write_dict_list(transactions, file_name)

def br():
	"""Prints a break line to show current record has changed"""
	safe_print("------------------", "\n")

def skipped_details_prompt(params, transaction, es_connection):
	"""Prompt the users to provide additional data"""

	safe_print("Base query: " + transaction["DESCRIPTION_UNMASKED"])
	store = enrich_transaction(params, transaction, es_connection, index=sys.argv[3])
	old_details = [store["PHYSICAL_MERCHANT"], store["STREET"], store["CITY"], string_cleanse(store["STATE"]), store["ZIP_CODE"],]
	old_details_formatted = ", ".join(old_details)
	safe_print("Old index details: {0}".format(old_details_formatted.encode("utf-8", "replace")))
	name = safe_input("What is the name of this merchant? \n")
	address = safe_input("At what address is this merchant located? \n")

	return name, address

def search_with_user_safe_input(params, es_connection, transaction):
	"""Search for a location by providing additional data"""

	index = sys.argv[4]
	prompt = "Base query: " + transaction["DESCRIPTION_UNMASKED"]
	prompt = prompt + " is insufficient to uniquely identify, please provide additional data \n"
	safe_print(prompt)

	# Give context to user
	safe_print("User " + str(transaction["UNIQUE_MEM_ID"]) + " Context: ")
	safe_print(params["compare_indices"]["user_context"][transaction["UNIQUE_MEM_ID"]], "\n")

	# Collect Additional Data
	store_name = safe_input("What is the name of the business? \n")
	store_address = safe_input("What is the store address with state, city and zip? \n")

	# Generate new query
	bool_search = get_bool_query(size=45)
	should_clauses = bool_search["query"]["bool"]["should"]

	name_query = get_qs_query(store_name, ["name"], boost=4)
	geo_query = get_qs_query(store_address, ["address", "locality", "region", "postcode"], boost=2)
	trans_query = get_qs_query(string_cleanse(transaction["DESCRIPTION_UNMASKED"]), ["_all"])
	should_clauses.append(trans_query)
	should_clauses.append(name_query)
	should_clauses.append(geo_query)

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

	safe_print("[enter] NULL")
	safe_print("{:7} Transaction did not occur at a specific location. Remove it from training data\n".format("[rm]"))

	user_input = safe_input("Please select a choice above: \n")

	# Remove from Set
	if user_input == "rm":
		return

	# Change factual_id, move to relinked
	if user_input in accepted_inputs:
		score, hit = get_hit(results, int(user_input))
		store["relinked_id"] = hit["factual_id"]
		params["compare_indices"]["relinked"].append(store)
	else:
		# Add transaction to another queue for later analysis
		params["compare_indices"]["NULL"].append(store)

def safe_input(prompt=""):
	"""Safely input a string"""

	try:
		result = input(prompt)
		return result
	except:
		return ""

def safe_print(string):
	"""Safely print a string"""

	try:
		print(string)
	except:
		print("Encoding Error. Continuing")

def decision_boundary(params, store, results):
	"""Decide if there is a match"""

	fields = ["name", "address", "locality", "region", "postcode"]
	old_details = [store["PHYSICAL_MERCHANT"], store["STREET"], store["CITY"], string_cleanse(store["STATE"]), store.get("ZIP_CODE", "")]
	accepted_inputs = [str(x) for x in list(range(5))]
	score, top_hit = get_hit(results, 0)

	# Add transaction back to the queue for later analysis if nothing found
	if score == False:
		safe_print("No matches found", "\n")
		params["compare_indices"]["skipped"].append(store)
		return

	# Compare Results
	old_details = [part.encode("utf-8") for part in old_details]
	new_details = [top_hit.get(field, "").encode("utf-8") for field in fields[0:5]]

	# Don't require input on matching details
	if new_details == old_details:
		safe_print("Record autolinked", "\n")
		user_input = "0"
	else:
		user_input = changed_id_user_prompt(params, old_details, results, store)

	# Remove is transaction isn't physical
	if user_input == "rm":
		return

	# Change factual_id, move to relinked
	if user_input in accepted_inputs:
		score, hit = get_hit(results, int(user_input))
		store["relinked_id"] = hit["factual_id"]
		params["compare_indices"]["relinked"].append(store)
	else:
		# Add transaction to another queue for later analysis
		params["compare_indices"]["skipped"].append(store)

def enrich_transaction(params, transaction, es_connection, index="", factual_id=""):
	"""Return a copy of a transaction, enriched with data from a 
	provided factual_id"""

	transaction = deepcopy(transaction)
	transaction["merchant_found"] = True
	fields_to_get = ["name", "region", "locality", "internal_store_number", "postcode", "address"]
	
	# Get merchant and suppress errors
	if factual_id == "":
		factual_id = transaction["factual_id"]

	with nostderr():
		merchant = get_merchant_by_id(params, factual_id, es_connection, index=index, fields=fields_to_get)

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

def find_merchant_by_address(params, store, es_connection, additional_data=[]):
	"""Match document with address to factual document"""

	fields = ["name^2", "address", "locality", "region", "postcode"]
	old_details = [store.get("PHYSICAL_MERCHANT", ""), store.get("STREET",""), store.get("CITY", ""), string_cleanse(store.get("STATE", "")), store.get("ZIP_CODE", "")]
	index = sys.argv[4]
	results = ""

	# Generate Query
	bool_search = get_bool_query(size=45)
	should_clauses = bool_search["query"]["bool"]["should"]

	# Additional Details
	if len(additional_data) > 0:
		for query in additional_data:
			if len(query) == 3:
				additional_query = get_qs_query(query[1], field_list=query[0], boost=query[2])
				should_clauses.append(additional_query)

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

def changed_id_user_prompt(params, old_details, results, store):
	"""Prompt a user for input to continue"""

	num_changed = len(params["compare_indices"]["id_changed"])
	num_relinked = len(params["compare_indices"]["relinked"])
	num_skipped = len(params["compare_indices"]["skipped"])
	total = num_changed + num_relinked + num_skipped
	percentage_relinked = num_relinked / total
	percentage_formatted = str(round(percentage_relinked * 100, 2)) + "%"
	old_details_formatted = [detail.decode("utf-8") for detail in old_details]
	old_details_formatted =  ", ".join(old_details_formatted)

	safe_print("Number id_changed remaining: ", num_changed)
	safe_print("Number relinked: ", num_relinked)
	safe_print("Number skipped: ", num_skipped, "\n")

	safe_print("DESCRIPTION_UNMASKED: {0}".format(store["DESCRIPTION_UNMASKED"]))
	safe_print("Query Sent: {0} \n".format(old_details_formatted.encode("utf-8", "replace")))
	
	for i in range(5):
		print_formatted_result(results, i)

	safe_print("[enter] None of the above")
	safe_print("[rm] Transaction did not occur at a specific location. Remove it from training data", "\n")
	user_input = ""
	user_input = safe_input("Please select a location, or press enter to skip: \n")

	return user_input

def print_formatted_result(results, index):
	"""Display a potential result in readable format"""

	fields_to_print = ["name", "address", "locality", "region", "postcode", "internal_store_number"]
	score, hit = get_hit(results, index)

	# No Result
	if not hit:
		safe_print("No hits found")
		return

	# Print Details
	details_formatted = [hit.get(field, "") for field in fields_to_print]
	details_formatted = ", ".join(details_formatted)
	safe_print("{:8}".format("[" + str(index) + "] ") + " " + str(details_formatted.encode("utf-8", "replace")))

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
		safe_print("Insufficient arguments. Please see usage")
		sys.exit()

	config = sys.argv[1]
	factual_list = sys.argv[2]

	config_included = config.endswith('.json')
	factual_list_included = factual_list.endswith('.txt')

	if not config_included  or not factual_list_included:
		safe_print("Erroneous arguments. Please see usage")
		sys.exit()

def add_local_params(params):
	"""Adds additional local params"""

	params["compare_indices"] = {}
	params["compare_indices"]["NULL"] = []
	params["compare_indices"]["id_changed"] = []
	params["compare_indices"]["details_changed"] = []
	params["compare_indices"]["skipped"] = []
	params["compare_indices"]["relinked"] = []
	params["compare_indices"]["user_context"] = collections.defaultdict(list)

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
#!/usr/local/bin/python3.3
# pylint: disable=all

"""This module takes a set of merchants with
their associated manually formatted store numbers.
The records are matched against Factual and if
a matching merchant is found the store numbers are
merged into the index. Merchants not found are saved
for further processing. This is a form of feature
engineering used to improve Meerkat's ability to label
transactions by location.

Created on March 25, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: Store numbers must be comma delimited

# python3.3 -m meerkat.merge_store_numbers [store_numbers_file]
# python3.3 -m meerkat.merge_store_numbers [store_numbers_directory]
# python3.3 -m meerkat.merge_store_numbers data/misc/Store\ Numbers/Clean/IAE

# Required Columns: 
# keywords: Name found in factual
# city: City name to match against
# store_number: Store numbers formated to match transactions
# address: Address to match against
# state: State to match against
# zip_code: Zip Code to match against

#####################################################

import csv 
import sys
import json
import os

from scipy.stats.mstats import zscore
from pprint import pprint
from elasticsearch import Elasticsearch, helpers

from meerkat.description_consumer import get_qs_query, get_bool_query

def load_store_numbers(file_name):
	"""Load Store Numbers from provided file"""

	print("LOADING: ", file_name)

	input_file = open(file_name, encoding="utf-8", errors='replace')
	stores = list(csv.DictReader(input_file, delimiter=","))
	input_file.close()

	for store in stores:
		if 'Keywords' in store:
			store['keywords'] = store.pop('Keywords')

	return stores

def z_score_delta(scores):
	"""Find the Z-Score Delta"""

	if len(scores) < 2:
		return None

	z_scores = zscore(scores)
	first_score, second_score = z_scores[0:2]
	z_score_delta = round(first_score - second_score, 3)

	return z_score_delta

def find_merchant(store):
	"""Match document with store number to factual document"""

	fields = ["address", "postcode", "name^1.5", "locality", "region"]
	search_parts = [store["address"], store["zip_code"][0:5], store["keywords"], store["city"], store["state"]]
	factual_id = ""
	top_result = ""

	# Generate Query
	bool_search = get_bool_query(size=45)
	should_clauses = bool_search["query"]["bool"]["should"]

	# Multi Field
	for i in range(len(fields)):
		sub_query = get_qs_query(search_parts[i], [fields[i]])
		should_clauses.append(sub_query)

	# Search Index
	results = search_index(bool_search)
	score, top_hit = get_hit(results, 0)

	if score == False:
		return "", ""

	# Allow User to Verify and Return 
	formatted = [top_hit.get("name", ""), top_hit.get("address", ""), top_hit.get("postcode", ""), top_hit.get("locality", ""), top_hit.get("region", ""),]
	formatted = ", ".join(formatted)
	print("Z-Score: ", score)
	print("Top Result: ", formatted.encode("utf-8"))
	print("Query Sent: ", search_parts)

	# Must Match Keywords
	if not (store["keywords"].lower() in top_hit["name"].lower()):
		return "", formatted	

	# Found a Match
	if score > 0.95:
		return top_hit["factual_id"], ""

	return factual_id, formatted

def get_hit(search_results, index):

	# Must have results
	if search_results['hits']['total'] == 0:
		return False, False

	hits = search_results['hits']['hits']
	scores = [hit['_score'] for hit in hits]
	z_score = z_score_delta(scores)
	hit = hits[index]

	return z_score, hit["_source"]

def update_merchant(factual_id, store):
	"""Update found merchant with store_number"""

	store_number = store["store_number"]
	body = {"doc" : {"internal_store_number" : store_number}}

	try:
		output_data = es_connection.update(index="factual_index", doc_type="factual_type", id=factual_id, body=body)
	except Exception:
		print("Failed to Update Merchant")

	status = True if output_data.get("_id", "") != "" else False

	return status

def search_index(query):
	"""Searches the merchants index and the merchant mapping"""

	input_data = json.dumps(query, sort_keys=True, indent=4\
	, separators=(',', ': ')).encode('UTF-8')
	output_data = ""

	try:
		output_data = es_connection.search(index="factual_index", body=query)
	except Exception:
		output_data = {"hits":{"total":0}}

	return output_data

def run(stores):
	"""Run the Program"""

	not_found = []
	total = len(stores)

	# Run Search
	for i in range(len(stores)):

		# Find Most Likely Merchant
		store = stores[i]
		factual_id, top_result = find_merchant(store)
		store['factual_id'] = factual_id
		store['top_result'] = top_result

		# Attempt to Update Document
		if len(factual_id) > 0:
			status = update_merchant(factual_id, store)
		else:
			print("Did Not Merge Store Number ", store["store_number"], " To Index", "\n")
			not_found.append(store)
			continue

		# Save Failed Attempts
		if status == False:
			print("Did Not Merge Store Number ", store["store_number"], " To Index")
			not_found.append(store)
		else:
			print("Successfully Merged Store Number:", store["store_number"], "into Factual Merchant:", factual_id, "\n")

	# Show Success Rate
	misses = len(not_found)
	hits = total - misses
	percent_merged = hits / total
	percent_missed = round((misses / total), 3)
	print("HITS: ", hits)
	print("MISSES: ", misses)
	print("PERCENT MERGED: ", percent_merged)

	# Save Not Found
	save_mapping(stores, percent_merged)
	#save_not_found(not_found, percent_merged)

def save_mapping(stores, percent_merged):
	"""Saves all results as a mapping file"""

	store_name = stores[0]['keywords']
	file_name = "/mnt/ephemeral/AggData_Factual_Merge/" + store_name + "_" + str(percent_merged * 100) + "%_success_rate" + ".csv"
	delimiter = ","
	output_file = open(file_name, 'w')
	dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=stores[0].keys())
	dict_w.writeheader()
	dict_w.writerows(stores)
	output_file.close()

def save_not_found(not_found, percent_merged):
	"""Save the stores not found in the index"""

	store_name = not_found[0]['keywords']
	file_name = "/mnt/ephemeral/AggData_Factual_Merge/" + store_name + "_" + str(percent_merged * 100) + "%_success_rate" + ".csv"
	delimiter = ","
	output_file = open(file_name, 'w')
	dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=not_found[0].keys())
	dict_w.writeheader()
	dict_w.writerows(not_found)
	output_file.close()

def process_multiple_merchants():
	"""Merge in all files within a provided folder"""

	dir_name = sys.argv[1]
	merchant_files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith(".csv")]

	# Process Merchants
	for merchant in merchant_files:

		try:
			stores = load_store_numbers(merchant)
			run(stores)
		except: 
			continue

def process_single_merchant():
	"""Merge in store numbers from a single files"""

	file_name = sys.argv[1]
	stores = load_store_numbers(file_name)
	run(stores)

def verify_arguments():
	"""Verify Usage"""

	sufficient_arguments = (len(sys.argv) == 2)

	if not sufficient_arguments:
		print("Insufficient arguments. Please see usage")
		sys.exit()

	# Single Merchant
	single_merchant = sys.argv[1].endswith('.csv')
	is_directory = os.path.isdir(sys.argv[1])

	if single_merchant:
		return

	# Directory of Merchants
	if not is_directory:
		print("Improper usage. Please provide a directory of csv files or a single csv")
		sys.exit()

	for f in os.listdir(sys.argv[1]):
		if f.endswith(".csv"):
			return

	# No CSV for Merchant Found
	print("Improper usage. Please provide at least one csv containing store numbers")
	sys.exit()

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""

	# Load and Process Merchants
	if os.path.isfile(sys.argv[1]):
		process_single_merchant()
	elif os.path.isdir(sys.argv[1]):
		process_multiple_merchants()

if __name__ == "__main__":

	verify_arguments()

	cluster_nodes = [
        "s01:9200",
        "s02:9200",
        "s03:9200",
        "s04:9200",
        "s05:9200",
        "s06:9200",
        "s07:9200",
        "s08:9200",
        "s09:9200",
        "s10:9200",
        "s11:9200",
        "s12:9200",
        "s13:9200",
        "s14:9200",
        "s15:9200",
        "s16:9200",
        "s17:9200",
        "s18:9200"
    ]

	es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True, sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)
	run_from_command_line(sys.argv)

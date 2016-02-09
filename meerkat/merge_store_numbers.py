#!/usr/local/bin/python3.3

"""This module takes a set of merchants with their associated manually
formatted store numbers. The records are matched against Factual and if a
matching merchant is found the store numbers are merged into the index.
Merchants not found are saved for further processing. This is a form of
feature engineering used to improve Meerkat's ability to label
transactions by location.

Created on Mar 25, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: Store numbers must be comma delimited
"""
python3 -m meerkat.merge_store_numbers [store_numbers_file] \
[cluster_address] [index] [doc_type]
python3 -m meerkat.merge_store_numbers [store_numbers_directory] \
[cluster_address] [index] [doc_type]
python3 -m meerkat.merge_store_numbers data/misc/Store\ Numbers/Clean/IAE \
127.0.0.1 factual_index factual_type
"""

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
import os
import logging

from scipy.stats.mstats import zscore
from elasticsearch import Elasticsearch
from queue import Queue
from threading import Thread

from meerkat.various_tools import get_bool_query, get_qs_query, string_cleanse

def load_store_numbers(file_name):
	"""Load Store Numbers from provided file"""

	logging.info("LOADING: {0}".format(file_name))

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

	fields = ["address", "postcode", "name", "locality", "region"]
	search_parts = [store["address"], store["zip_code"][0:5],\
	 store["keywords"], store["city"], string_cleanse(store["state"])]
	factual_id = ""

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
		return "", "", ""

	# Allow User to Verify and Return
	formatted = [top_hit.get("name", ""), top_hit.get("address", ""),\
	 top_hit.get("postcode", ""), top_hit.get("locality", ""), \
	 top_hit.get("region", ""),]
	formatted = ", ".join(formatted)

	message = """
		File: {0}
		Z-Score: {1}
		Top Result: {2}
		Query Sent: {3}
		""".format(store['keywords'], score, formatted.encode("utf-8"), search_parts)
	# logging.warning(message)

	# Must Match Keywords
	if not store["keywords"].lower() in top_hit["name"].lower():
		return "", formatted, ""

	# Found a Match
	if score > 0.95:
		return top_hit["factual_id"], "", ""

	return factual_id, formatted, message

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
		output_data = get_es_connection().update(index=sys.argv[3], \
		doc_type=sys.argv[4], id=factual_id, body=body)
	except Exception:
		logging.warning("Failed to Update Merchant")

	status = True if output_data.get("_id", "") != "" else False

	return status

def search_index(query):
	"""Searches the merchants index and the merchant mapping"""

	output_data = ""

	try:
		output_data = get_es_connection().search(index=sys.argv[3], body=query)
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
		factual_id, top_result, message = find_merchant(store)
		store['factual_id'] = factual_id
		store['top_result'] = top_result

		# Attempt to Update Document
		if len(factual_id) > 0:
			status = update_merchant(factual_id, store)
		else:
			logging.warning("{1}Did Not Merge Store Number {0} to Index\n".\
			format(store["store_number"], message))
			not_found.append(store)
			continue

		# Save Failed Attempts
		if status == False:
			logging.warning("{1}Did Not Merge Store Number {0} to Index".\
			format(store["store_number"], message))
			not_found.append(store)
		else:
			logging.warning("{2}Successfully Merged Store Number: {0} into Factual Merchant: {1}\n".\
			format(store["store_number"], factual_id, message))

	# Show Success Rate
	misses = len(not_found)
	hits = total - misses
	percent_merged = hits / total
	#percent_missed = round((misses / total), 3)
	logging.warning("HITS: {0}".format(hits))
	logging.warning("MISSES: {0}".format(misses))
	logging.warning("PERCENT MERGED: {0}".format(percent_merged))

	# Save Mapping
	save_mapping(stores, percent_merged)

def save_mapping(stores, percent_merged):
	"""Saves all results as a mapping file"""

	store_name = stores[0]['keywords']

	path = "data/output/AggData_Factual_Merge/"
	os.makedirs(path, exist_ok=True)

	file_name = path + store_name + "_" + \
	str(percent_merged * 100) + "%_success_rate" + ".csv"
	delimiter = ","
	output_file = open(file_name, 'w')
	dict_w = csv.DictWriter\
	(output_file, delimiter=delimiter, fieldnames=stores[0].keys())
	dict_w.writeheader()
	dict_w.writerows(stores)
	output_file.close()

def start_thread(_queue):
	while True:
		merchant = _queue.get()
		stores = load_store_numbers(merchant)
		run(stores)
		_queue.task_done()

def process_multiple_merchants():
	"""Merge in all files within a provided folder"""

	dir_name = sys.argv[1]
	merchant_files = [os.path.join(dir_name, f) \
	for f in os.listdir(dir_name) if f.endswith(".csv")]

	_queue = Queue(maxsize=0)
	num_threads = 12

	for i in range(num_threads):
		worker = Thread(target=start_thread, args=(_queue,))
		worker.name = "Thread %d" % i
		worker.setDaemon(True)
		worker.start()

	# Process Merchants
	for merchant in merchant_files:
		logging.warning("Processing %s" % merchant)
		try:
			_queue.put(merchant)
		except: 
			continue

	_queue.join()

def process_single_merchant():
	"""Merge in store numbers from a single files"""

	file_name = sys.argv[1]
	stores = load_store_numbers(file_name)
	run(stores)

def verify_arguments():
	"""Verify Usage"""

	sufficient_arguments = (len(sys.argv) == 5)

	if not sufficient_arguments:
		logging.warning("Insufficient arguments. Please see usage")
		sys.exit()

	# Single Merchant
	single_merchant = sys.argv[1].endswith('.csv')
	is_directory = os.path.isdir(sys.argv[1])

	if single_merchant:
		return

	# Directory of Merchants
	if not is_directory:
		logging.warning\
		("Improper usage. Please provide a directory of csv files or a single csv")
		sys.exit()

	for filename in os.listdir(sys.argv[1]):
		if filename.endswith(".csv"):
			return

	# No CSV for Merchant Found
	logging.warning\
	("Improper usage. Please provide at least one csv containing store numbers")
	sys.exit()

def run_from_command_line():
	"""Runs these commands if the module is invoked from the command line"""

	# Load and Process Merchants
	if os.path.isfile(sys.argv[1]):
		process_single_merchant()
	elif os.path.isdir(sys.argv[1]):
		process_multiple_merchants()

def get_es_connection():
	""" Get the Elastic search connection. """
	es_connection = Elasticsearch([sys.argv[2]], sniff_on_start=True, 
		sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)
	return es_connection

if __name__ == "__main__":
	verify_arguments()
	
	logging.basicConfig(format='%(threadName)s %(asctime)s %(message)s',
		filename='merging.log', level=logging.WARNING)

	logging.warning("Beginning log...")

	run_from_command_line()

	logging.shutdown()

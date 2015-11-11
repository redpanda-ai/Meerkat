#!/usr/local/bin/python3.3

"""This script takes a list of transactions and their matched factual_id's
and populates the input file with additional fields from the factual index

Created on July 15, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: In Progress
# python3.3 -m meerkat.labeling_tools.get_merchants_by_ID config/test.json data/misc/ground_truth_card.txt

# Required Params:

#####################################################

import os
import sys

from meerkat.various_tools import (load_params, get_es_connection, \
get_merchant_by_id, load_dict_list, write_dict_list)

def enrich_transactions(params, es_connection):
	"""Enrich a set of transactions using a provided factual_id"""

	transactions = load_dict_list(sys.argv[2])

	for transaction in transactions:

		merchant = get_merchant_by_id(params,\
		 transaction["FACTUAL_ID"], es_connection)

		# No merchant found for factual_id
		if merchant == None:
			continue

		# Enrich 
		location = merchant.get\
		("pin", {}).get("location", {}).get("coordinates", ["", ""])
		latitude = location[1]
		longitude = location[0]
		transaction["PHYSICAL_MERCHANT"] = merchant.get("name", "")
		transaction["STORE_NUMBER"] = merchant.get("internal_store_number", "")
		transaction["CITY"] = merchant.get("locality", "")
		transaction["STATE"] = merchant.get("region", "")
		transaction["LATITUDE"] = latitude
		transaction["LONGITUDE"] = longitude
		transaction["STREET"] = merchant.get("address", "")

	file_name = "data/misc/" + \
	os.path.splitext(os.path.basename(sys.argv[1]))[0] + "_enriched.txt"
	write_dict_list(transactions, file_name)

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

def run_from_command_line():
	"""Runs these commands if the module is invoked from the command line"""

	verify_arguments()
	params = load_params(sys.argv[1])
	es_connection = get_es_connection(params)
	enrich_transactions(params, es_connection)
	
if __name__ == "__main__":
	run_from_command_line()

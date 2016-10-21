import json
import argparse
import csv
import re
import logging
import pandas as pd
import sys
import yaml
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

#from meerkat.various_tools import load_params

logging.config.dictConfig(yaml.load(open('meerkat/file_processing/logging.yaml', 'r')))
logger = logging.getLogger('basic')

def get_renames(rename_type):
	"""Fetches the proper rename dict, used to rename columns in the dataframe."""
	renames = None
	if rename_type in ['credit', 'debit_ach']:
		return {
			"row_id": "transaction_id",
			"at_transactiondescription2": "description",
			"at_transactioncity": "city_or_phone",
			"at_transactionstateprovince": "state",
			"at_transactioncountrycode": "country",
			"at_transactionpostalcode1": "postal_code"
		}
	elif rename_type == 'debit_pos':
		return {
			"row_id": "transaction_id",
			"card_acpt_merchant_name": "description",
			"at_transactioncity": "city_or_phone",
			"at_transactionstateprovince": "state",
			"at_transactioncountrycode": "country",
			"at_transactionpostalcode1": "postal_code"
		}
	else:
		logger.error("{0} is Not a valid file_type, aborting".format(rename_type))
		sys.exit()
	return renames

def get_file_type(args):
	"""Determines the file type, based upon the name of the input file."""
	logger.warning(args.input_file)
	if args.input_file.find('_credit') != -1:
		file_type = 'credit'
	elif args.input_file.find('_debit_pos') != -1:
		file_type = 'debit_pos'
	elif args.input_file.find('_debit_ach') != -1:
		file_type = 'debit_ach'
	else:
		logger.error("Cannot identify file type for {0}, aborting".format(args.file_name))
		sys.exit()
	logger.info("File type is {0}".format(file_type))
	return file_type

def preprocess_dataframe(args):
	"""Reads the input_file into a dataframe"""
	kwargs = {
		"quoting": csv.QUOTE_NONE, "encoding": "utf-8", "sep": "|", "error_bad_lines": True,
		"warn_bad_lines": True, "chunksize": 1, "na_filter": False
	}
	#We don't really need the entire file get 1 row from the first chunk
	reader = pd.read_csv(args.input_file, **kwargs)
	my_df = reader.get_chunk(0)
	header_names = list(my_df.columns.values)
	#logger.info(my_df)
	logger.info(header_names)
	#Set all data types to "str"
	dtype = {}
	for column in header_names:
		dtype[column] = "str"
	#logger.info(dtype)
	#Now lets grab the entire file as a dataframe
	del kwargs["chunksize"]
	kwargs["dtype"] = dtype
	return pd.read_csv(args.input_file, **kwargs)

def clean_dataframe(my_df, renames):
	"""Removes unneeded columns, renames others."""
	my_df.rename(index=str, columns=renames, inplace=True)
	header_names = list(my_df.columns.values)
	reverse_renames = {}
	for key in renames.keys():
		val = renames[key]
		reverse_renames[val] = key
	#Remove unused columns
	for column in header_names:
		if column not in reverse_renames:
			logger.info("Removing superflous column {0}".format(column))
			del my_df[column]
	#lambda functions
	phone_regex = "^[0-9][0-9\-]*$"
	website_regex = "^.*\.com.*$"
	city_regex = "^[a-z ]+.*$"
	get_phone = lambda x: x["city_or_phone"] \
		if re.match(phone_regex, x["city_or_phone"]) else ""
	get_website = lambda x: x["city_or_phone"] \
		if re.match(website_regex, x["city_or_phone"], flags=re.IGNORECASE) else ""
	get_city = lambda x: x["city_or_phone"] \
		if ( re.match(city_regex, x["city_or_phone"], flags=re.IGNORECASE) and not re.match(website_regex, x["city_or_phone"], flags=re.IGNORECASE) ) else ""
	#Add some columns
	my_df["phone_number"] = my_df.apply(get_phone, axis=1)
	my_df["city"] = my_df.apply(get_city, axis=1)
	my_df["website_url"] = my_df.apply(get_website, axis=1)
	#Remove processed column
	del my_df["city_or_phone"]

def get_rnn_merchant(my_df):
	#from meerkat.classification.load_model import get_tf_rnn_by_path
	#merchant_rnn = get_tf_rnn_by_path("rnn/bilstm.ckpt", "rnn/w2i.json")
	#predicted = merchant_rnn([{"Description": my_df["description"]}])[0]["Predicted"]
	predicted = "RNN_MERCHANT FIXME"
	return predicted

def get_store_number(my_df):
	#merchant_rnn = get_tf_rnn_by_path("rnn/bilstm.ckpt", "rnn/w2i.json")
	#predicted = merchant_rnn([{"Description": my_df["description"]}])[0]["Predicted"]
	predicted = "STORE_NUMBER FIXME"
	return predicted

def get_blank_transaction_dict():
	return {
		"row_id": [], "merchant_name": [], "address": [], "city": [],
		"state": [], "zip_code": [], "phone": [], "longitude": [],
		"latitude": [], "website_url": [], "store_number": [] }

def get_results_df_from_web_service(my_web_request, container):
	"""I do stuff"""
	#logger.info("Number of transactions: {0}".format(len(my_web_request["transaction_list"])))
	response = requests.post("https://localhost:443/meerkat_datadeal/", verify=False, data=json.dumps(my_web_request))
	#logger.info(json.dumps(my_web_request, sort_keys=True, indent=4, separators=(',', ':')))
	if "data" not in response.text:
		logger.critical("There is no data in the response, the response is {0}".format(response))
		sys.exit()
	response_data = json.loads(response.text)["data"]
	#logger.info(json.dumps(response_data, sort_keys=True, indent=4, separators=(',', ':')))
	my_results = get_blank_transaction_dict()
	my_keys = my_results.keys()
	for transaction in response_data["transaction_list"]:
		for key in my_keys:
			my_results[key].append(transaction[key])

	#Add non-transaction_list values
	my_results["cobrand_id"] = 0
	my_results["user_id"] = 0
	my_results["container"] = container

	#Convert the dictionary to a dataframe
	my_df = pd.DataFrame.from_dict(my_results)
	#logger.info("my_df:\n{0}".format(my_df))
	return my_df
	#sys.exit()

def main_process(args=None):
	"""Opens up the input file and loads it into a dataframe"""
	if args is None:
		args = parse_arguments(sys.argv[1:])
	logger.info("Starting main process")
	my_df = preprocess_dataframe(args)
	file_type = get_file_type(args)
	renames = get_renames(file_type)
	my_df.rename(index=str, columns=renames, inplace=True)
	clean_dataframe(my_df, renames)
	my_df["RNN_merchant_name"] = my_df.apply(get_rnn_merchant, axis=1)
	my_df["store_number"] = my_df.apply(get_store_number, axis=1)
	#the web service input requires ledger_entry, amount, and container"

	#make a payload, convert the dataframe
	amount, ledger_entry, container, my_date = 10, "debit", "bank", "2016-01-01"
	if file_type == "credit":
		container = "card"
	my_transactions = my_df.to_dict(orient="records")

	transaction_max = 100
	transaction_count = 0
	result_dfs = []
	my_web_request = None
	for transaction in my_transactions:
		#Create a new batch
		if transaction_count % transaction_max == 0:
			#Append existing batch to web_requests
			if my_web_request is not None:
				logger.info("Transaction count {0}".format(transaction_count))
				result_dfs.append(get_results_df_from_web_service(my_web_request, container))
				#web_requests.append(my_web_request.copy())
			#Create a new batch
			my_web_request = {
				"cobrand_id": 0,
				"user_id": 0,
				"container": container,
				"transaction_list": []
			}
		#Ensure that each transaction has some default values
		transaction["transaction_id"] = int(transaction["transaction_id"])
		transaction["ledger_entry"] = "debit"
		transaction["amount"] = 10
		transaction["date"] = my_date
		#Add each transaction to the transaction_list for the web_request
		my_web_request["transaction_list"].append(transaction.copy())
		transaction_count += 1
	#Close the final batch of transactions
	if my_web_request is not None:
		result_dfs.append(get_results_df_from_web_service(my_web_request, container))
		#web_requests.append(my_web_request.copy())

	results_df = pd.concat(result_dfs, ignore_index=True)
	logger.info("All Results: {0}".format(results_df.shape))
	header = [ "row_id", "merchant_name", "address", "city", "state", "zip_code",
		"phone", "longitude", "latitude", "website_url", "store_number" ]
	df_column_list = list(results_df.columns.values)
	for column in df_column_list:
		if column not in header:
			del results_df[column]

	results_df.to_csv("test.csv", index=header, sep="|", mode="w", header=header)
	logger.info("Written to test.csv")
	#logger.info("All Responses: {0}".format(result_dfs))
	

	#logger.info(json.dumps(web_requests[len(web_requests) - 1], sort_keys=True, indent=4, separators=(',', ':')))
	#logger.info("Web request count: {0}".format(len(web_requests)))

	#Turn the dataframe into batches of transactions as JSON
	#Set the web reqest to the web service
	#Parse the results
	#Write results to file

	#logger.info(my_df)
	#I should call the web service with the dataframe broken into 1000 transactions per call.
	#Take all of the responses and write it out to a file...
	logger.info("All done.")

def parse_arguments(args):
	"""Correctly parses command line arguments for the program"""
	parser = argparse.ArgumentParser(description="It's simple.")
	#Required arguments
	parser.add_argument('input_file', help='Path to input file on local drive')
	return parser.parse_args(args)

if __name__ == "__main__":
	main_process()



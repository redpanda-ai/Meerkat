"""This module does file processing for the data-deal branch of Meerkat"""
import json
import argparse
import re
import logging
import sys
import pandas as pd
import yaml
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

#from meerkat.various_tools import load_params
from meerkat.classification.load_model import get_tf_rnn_by_path
MODEL_PATH = "meerkat/classification/models/multi_rnn_model/"
MULTICLASS_RNN = get_tf_rnn_by_path(MODEL_PATH + "bilstm.ckpt", MODEL_PATH + "w2i.json")

logging.config.dictConfig(yaml.load(open('meerkat/file_processing/logging.yaml', 'r')))
LOGGER = logging.getLogger('basic')

def get_renames(rename_type):
	"""Fetches the proper rename dict, used to rename columns in the dataframe."""
	renames = None
	if rename_type in ['credit', 'debit_ach', 'debit_pos']:
		return {
			"row_id": "transaction_id",
			"at_transactiondescription2": "description",
			"at_transactioncity": "mystery_field",
			"at_transactionstateprovince": "state",
			"at_transactioncountrycode": "country",
			"at_transactionpostalcode1": "postal_code"
		}
	else:
		LOGGER.error("{0} is Not a valid file_type, aborting".format(rename_type))
		sys.exit()
	return renames

def get_file_type(args):
	"""Determines the file type, based upon the name of the input file."""
	LOGGER.warning(args.input_file)
	if args.input_file.find('_credit') != -1:
		file_type = 'credit'
	elif args.input_file.find('_debit_pos') != -1:
		file_type = 'debit_pos'
	elif args.input_file.find('_debit_ach') != -1:
		file_type = 'debit_ach'
	else:
		LOGGER.error("Cannot identify file type for {0}, aborting".format(args.file_name))
		sys.exit()
	LOGGER.info("File type is {0}".format(file_type))
	return file_type

def preprocess_dataframe(args):
	"""Reads the input_file into a dataframe"""
	kwargs = {
		"encoding": "utf-8", "sep": "|", "error_bad_lines": True,
		"warn_bad_lines": True, "chunksize": 1, "na_filter": False
	}
	#We don't really need the entire file get 1 row from the first chunk
	reader = pd.read_csv(args.input_file, **kwargs)
	my_df = reader.get_chunk(0)
	header_names = list(my_df.columns.values)
	#Set all data types to "str"
	dtype = {}
	for column in header_names:
		dtype[column] = "str"
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
			LOGGER.info("Removing superflous column {0}".format(column))
			del my_df[column]
	#lambda functions
	phone_regex = r"^[0-9][0-9\-]*$"
	website_regex = r"^.*\.com.*$"
	city_regex = r"^[a-z ]+.*$"
	#split 'mystery_field' into phone, website, or city
	get_phone = lambda x: x["mystery_field"] \
		if re.match(phone_regex, x["mystery_field"]) else ""
	get_website = lambda x: x["mystery_field"] \
		if re.match(website_regex, x["mystery_field"], flags=re.IGNORECASE) else ""
	get_city = lambda x: x["mystery_field"] \
		if (re.match(city_regex, x["mystery_field"], flags=re.IGNORECASE) \
		and not re.match(website_regex, x["mystery_field"], flags=re.IGNORECASE)) else ""
	#Add some columns
	my_df["phone_number"] = my_df.apply(get_phone, axis=1)
	my_df["city"] = my_df.apply(get_city, axis=1)
	my_df["website_url"] = my_df.apply(get_website, axis=1)
	#Remove processed column
	del my_df["mystery_field"]

def has_numbers(inputString):
	"""Check if input string has number"""
	return any(char.isdigit() for char in inputString)

def process_description(my_df):
	"""Preprocess description field"""
	merchant = my_df["description"]

	# Find the following patterns and replace with empty string
	merchants = ["ABC","PAYPAL", "SQ", "GOOGLE", "MSFT", "MICROSOFT", "IN", "VESTA"]
	patterns = [r"" + m + r" +\*" for m in merchants]
	patterns.append(r"EEPAY/GARNWC")
	patterns.append(r"ABC\*")
	patterns.append(r"USA\*")
	for cur_pattern in patterns:
		pattern = re.compile(cur_pattern, re.IGNORECASE)
		merchant = pattern.sub("", merchant)
	return merchant

def run_multi_class_rnn(my_df):
	"""Get multi-class RNN result"""
	desc = my_df["description"]
	print(desc)
	if desc == "":
		return {}
	tagged = MULTICLASS_RNN([{"Description": desc}])
	print(desc)
	return tagged[0]["Predicted"]

def get_rnn_merchant(my_df):
	"""This is a stub implementation, no multi-class RNN exists."""
	desc = my_df["description"]
	predicted = my_df["predicted"]
	LOGGER.info(predicted)
	if "merchant" in predicted:
		merchant_str = " ".join(predicted["merchant"])
		predicted["merchant"] = [merchant_str]
		tag = re.match(re.escape(predicted["merchant"][0]),\
			my_df["description"], re.IGNORECASE)
		if tag: #There are matches
			return desc[tag.start():tag.end()]
		else:
			return predicted["merchant"][0]
	else:
		return ""

def get_store_number(my_df):
	"""This is a stub implementation, no multi-class RNN exists."""
	return my_df["predicted"].get("store_number", [])

def get_results_df_from_web_service(my_web_request, container):
	"""Sends a single web request dict to the web service, then converts the
	result into a results dataframe"""
	response = requests.post("https://localhost:443/meerkat_datadeal/", verify=False,
		data=json.dumps(my_web_request))
	if "data" not in response.text:
		LOGGER.critical("There is no data in the response, the response is {0}".format(response))
		sys.exit()
	response_data = json.loads(response.text)["data"]
	#Create a blank dictionary of results, each value needs to be a list
	my_results = {
		"row_id": [], "input_description": [], "merchant_name": [],
		"description_substring": [], "address": [], "city": [],
		"state": [], "zip_code": [], "phone": [], "longitude": [],
		"latitude": [], "website_url": [], "store_number": []}
	my_keys = my_results.keys()
	#Append an element to for each key in each transaction
	for transaction in response_data["transaction_list"]:
		for key in my_keys:
			my_results[key].append(transaction[key])
	#Add non-transaction_list values
	my_results["cobrand_id"] = 0
	my_results["user_id"] = 0
	my_results["container"] = container
	#Convert the dictionary to a dataframe
	my_df = pd.DataFrame.from_dict(my_results)
	return my_df

def process_postal_code(my_df):
	"""Preprocess postal_code field"""
	postal_code = my_df["postal_code"]
	if '.' in postal_code:
		postal_code = postal_code.split('.')[0]
	if '-' in postal_code:
		postal_code = postal_code.split('-')[0]
	if postal_code == '00000':
		return postal_code
	postal_code = postal_code.zfill(5)
	return postal_code if postal_code != '00000' else ''

def main_process(args=None):
	"""Opens up the input file and loads it into a dataframe"""
	if args is None:
		args = parse_arguments(sys.argv[1:])
	LOGGER.info("Starting main process")
	#1. Get the input data
	my_df = preprocess_dataframe(args)
	file_type = get_file_type(args)
	renames = get_renames(file_type)
	#2. Rename certain columns in the dataframe
	my_df.rename(index=str, columns=renames, inplace=True)
	#3. Remove unneeded columns, split mystery_field
	clean_dataframe(my_df, renames)
	my_df["postal_code"] = my_df.apply(process_postal_code, axis=1)
	my_df["input_description"] = my_df["description"]
	my_df["description"] = my_df.apply(process_description, axis=1)
	#4. Use the RNN to grab a few more columns
	my_df["predicted"] = my_df.apply(run_multi_class_rnn, axis=1)
	my_df["RNN_merchant_name"] = my_df.apply(get_rnn_merchant, axis=1)
	my_df["store_number"] = my_df.apply(get_store_number, axis=1)
	del my_df["predicted"]

	container = "bank"
	if file_type == "credit":
		container = "card"
	#5. Transform the dataframe to a record-oriented dictionary
	my_transactions = my_df.to_dict(orient="records")

	transaction_max, transaction_count, result_dfs = 100, 0, []
	my_web_request = None
	#6. Create batches of transactions to send as a data payload, and issue web service
	#requests.
	for transaction in my_transactions:
		#Create a new batch
		if transaction_count % transaction_max == 0:
			#Append existing batch to web_requests
			if my_web_request is not None:
				LOGGER.info("Transaction count {0}".format(transaction_count))
				result_dfs.append(get_results_df_from_web_service(my_web_request, container))
			#Create a new batch
			my_web_request = {"cobrand_id": 0, "user_id": 0, "container": container,
				"transaction_list": []}
		#Ensure that each transaction has some default values
		transaction["transaction_id"] = int(transaction["transaction_id"])
		transaction["ledger_entry"] = "debit"
		transaction["amount"] = 10
		transaction["date"] = "2016-01-01"
		if file_type == "debit_ach":
			my_web_request["services_list"] = ["CNN"]
		#Add each transaction to the transaction_list for the web_request
		my_web_request["transaction_list"].append(transaction.copy())
		transaction_count += 1
	#Close the final batch of transactions
	if my_web_request is not None:
		result_dfs.append(get_results_df_from_web_service(my_web_request, container))

	#7. Merge all results into a single dataframe
	results_df = pd.concat(result_dfs, ignore_index=True)
	LOGGER.info("All Results: {0}".format(results_df.shape))
	header = ["row_id", "input_description", "merchant_name", "description_substring",
		 "address", "city", "state", "zip_code",
		"phone", "longitude", "latitude", "website_url", "store_number"]
	#8. Drop extraneous columns
	df_column_list = list(results_df.columns.values)
	for column in df_column_list:
		if column not in header:
			del results_df[column]
	#9. Re-order everything
	results_df = results_df[header]
	#10. Write out the results into a delimited file
	results_df.to_csv(args.output_file, index=False, sep="|", mode="w", header=header)
	LOGGER.info("Results written to {0}".format(args.output_file))

def parse_arguments(args):
	"""Correctly parses command line arguments for the program"""
	parser = argparse.ArgumentParser(description="It's simple.")
	#Required arguments
	parser.add_argument('input_file', help='Path to input file on local drive')
	parser.add_argument('output_file', help='Path to output file on local drive')
	return parser.parse_args(args)

if __name__ == "__main__":
	LOGGER.info("Starting main_process.")
	main_process()
	LOGGER.info("main_process complete.")



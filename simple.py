import argparse
import csv
import re
import logging
import pandas as pd
import sys
import yaml

from meerkat.various_tools import load_params
from meerkat.classification.load_model import get_tf_rnn_by_path

logging.config.dictConfig(yaml.load(open('logging.yaml', 'r')))
logger = logging.getLogger('basic')

merchant_rnn = get_tf_rnn_by_path("rnn/bilstm.ckpt", "rnn/w2i.json")

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
			"at_transactionpostal_code1": "postal_code"
		}
	elif rename_type == 'debit_pos':
		return {
			"row_id": "transaction_id",
			"card_acpt_merchant_name": "description",
			"at_transactioncity": "city_or_phone",
			"at_transactionstateprovince": "state",
			"at_transactioncountrycode": "country",
			"at_transactionpostal_code1": "postal_code"
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
	#We don't really need the entire file get 2 rows from the first chunk
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
	get_phone = lambda x: x["city_or_phone"] \
		if re.match("^[0-9\-]*$", x["city_or_phone"]) else ""
	get_city = lambda x: x["city_or_phone"] \
		if not re.match("^[0-9\-]*$", x["city_or_phone"]) else ""
	#Add some columns
	my_df["phone"] = my_df.apply(get_phone, axis=1)
	my_df["city"] = my_df.apply(get_city, axis=1)
	#Remove processed column
	del my_df["city_or_phone"]

def get_rnn_merchant(my_df):
	predicted = merchant_rnn([{"Description": my_df["description"]}])[0]["Predicted"]
	return predicted
	#try:
	#	result = re.sub(re.escape(predicted, "", my_df["description"],
	#		flags=re.IGNORECASE))
	#	result = my_df["description"][result.start():result.end()]
	#except:
	#	result = ""
	#return result

def main_process(args=None):
	"""Opens up the input file and loads it into a dataframe"""
	if args is None:
		args = parse_arguments(sys.argv[1:])
	logger.info("Starting main process")
	my_df = preprocess_dataframe(args)
	renames = get_renames(get_file_type(args))
	my_df.rename(index=str, columns=renames, inplace=True)
	clean_dataframe(my_df, renames)
	#get_rnn_merchant = lambda x: merchant_rnn([{"Description": x["description"]}][0]["Predicted"])
	my_df["rnn_merchant"] = my_df.apply(get_rnn_merchant, axis=1)
#	apply_rnn(my_df)
	logger.info(my_df)

def parse_arguments(args):
	"""Correctly parses command line arguments for the program"""
	parser = argparse.ArgumentParser(description="It's simple.")
	#Required arguments
	parser.add_argument('input_file', help='Path to input file on local drive')
	return parser.parse_args(args)

if __name__ == "__main__":
	main_process()



"""This module will sync local agg data with s3"""

import argparse
import json
import logging
import logging.config
import sys
import yaml

from meerkat.various_tools import validate_configuration
#:wfrom .get_agg_data import main_process as get_agg_data
#from .get_agg_data import parse_arguments as parse_agg_arguments
#from .get_merchant_dictionaries import parse_arguments as parse_merchant_arguments

import meerkat.geomancer.get_merchant_dictionaries as merchant_dictionaries
import meerkat.geomancer.get_agg_data as agg_data


logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('__main__')

def parse_arguments(args):
	"""Parse arguments from command line"""
	parser = argparse.ArgumentParser()
	#FIXME: Would be nice if we could validate this
	parser.add_argument("config_file", help="file used to set parameters")
	args = parser.parse_args(args)
	return args

def get_module_args(parse_function, config_object):
	arg_list = []
	for item in config_object.keys():
		logger.info("Key: {0}, Value: {1}".format(item, config_object[item]))
		arg_list.append(item)
		arg_list.append(config_object[item])
	logger.info(arg_list)
	return parse_function(arg_list)

def main_process():
	"""Execute the main programe"""
	logger.info("Starting")
	logger.info("Parsing arguments.")
	args = parse_arguments(sys.argv[1:])
	logger.info("Validating configuration")
	config = validate_configuration(args.config_file, "meerkat/geomancer/config/schema.json")

	#Get AggData
	logger.info("Fetching agg_data")
	module = agg_data
	sub_config = config["agg_data"]
	module.Worker(sub_config).main_process()

	#Get Merchant dictionaries
	logger.info("Getting merhcant dictionaries")
	module = merchant_dictionaries
	sub_config = config["merchant_dictionaries"]
	module.Worker(sub_config).main_process()

if __name__ == "__main__":
	main_process()


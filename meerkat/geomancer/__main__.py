"""This module will sync local agg data with s3"""

import argparse
import json
import logging
import logging.config
import sys
import yaml

from meerkat.various_tools import validate_configuration
from .get_agg_data import main_process as get_agg_data
from .get_agg_data import parse_arguments as parse_agg_arguments

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('__main__')

def parse_arguments(args):
	"""Parse arguments from command line"""
	parser = argparse.ArgumentParser()
	parser.add_argument("config_file", help="file used to set parameters")
	args = parser.parse_args(args)
	return args

def main_process():
	"""Execute the main programe"""
	logger.info("Starting")
	logger.info("Parsing arguments.")
	args = parse_arguments(sys.argv[1:])
	logger.info("Validating configuration")
	config = validate_configuration(args.config_file, "meerkat/geomancer/config/schema.json")
	#Convert config["agg_data_to"] to list
	#pre_list_dict = config["agg_data"]
	#logger.info("Agg_data {0}".format(pre_list_dict))
	my_list = []
	for item in config["agg_data"].keys():
		logger.info("Key: {0}, Value: {1}".format("--" + item, config["agg_data"][item]))
		my_list.append("--" + item)
		my_list.append(config["agg_data"][item])
	logger.info(my_list)
	agg_data_args = parse_agg_arguments(my_list)
	get_agg_data(agg_data_args)

if __name__ == "__main__":
	main_process()


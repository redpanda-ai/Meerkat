"""This module will sync local agg data with s3"""

import argparse
import json
import logging
import logging.config
import sys
import yaml

from meerkat.various_tools import validate_configuration

import meerkat.geomancer.get_merchant_dictionaries as merchant_dictionaries
import meerkat.geomancer.get_agg_data as agg_data
import meerkat.geomancer.pybossa.build_pybossa_project as builder
import meerkat.geomancer.interrogate as interrogate

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
	logger.info("Validating configuration!!!")
	config = validate_configuration(args.config_file, "meerkat/geomancer/config/schema.json")

	module_list = [("agg_data", agg_data), ("merchant_dictionaries", merchant_dictionaries),
		("interrogate", interrogate), ("pybossa_project", builder)]
	for name, module in module_list:
		if name in config:
			logger.info("Activating {0} module.".format(name))
			config_snippet = config[name]
			module.Worker(config_snippet).main_process()

if __name__ == "__main__":
	main_process()


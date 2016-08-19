"""This module will sync local agg data with s3"""

import argparse
import json
import logging
import logging.config
import sys
import yaml

from meerkat.various_tools import validate_configuration

import meerkat.geomancer.get_merchant_dictionaries as merchant_dictionaries
import meerkat.geomancer.pybossa.build_pybossa_project as builder
import meerkat.geomancer.interrogate as interrogate
import meerkat.geomancer.get_top_merchant_data as top_merchant_data

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
	"""Execute the main program"""
	logger.info("Starting")
	logger.info("Parsing arguments.")
	args = parse_arguments(sys.argv[1:])

	logger.info("Validating common configuration")
	common_config = validate_configuration("meerkat/geomancer/config/config_common.json",
		"meerkat/geomancer/config/schema.json")
	logger.info("Valid common configuration")
	common_config_snippet = common_config["common_config"]

	logger.info("Validating configuration")
	config = validate_configuration(args.config_file,
		"meerkat/geomancer/config/schema.json")
	logger.info("Valid configuration")

	module_list = [merchant_dictionaries, builder, top_merchant_data, interrogate]
	for module in module_list:
		if module.Worker.name in config:
			logger.info("Activating {0} module.".format(module.Worker.name))
			config_snippet = config[module.Worker.name]
			common_config_snippet = module.Worker(common_config_snippet, config_snippet).main_process()
			logger.info(common_config_snippet)

if __name__ == "__main__":
	main_process()


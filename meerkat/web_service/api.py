"""This module defines the Meerkat web service API."""
import concurrent.futures
import json
import logging
import os
import re
import shutil

from tornado import gen
from tornado_json.requesthandlers import APIHandler

from meerkat.web_service.web_consumer import WebConsumer
from meerkat.web_service import schema
from meerkat.various_tools import (load_params, get_us_cities,\
	load_hyperparameters)

class Meerkat_API(APIHandler):
	"""This class is the Meerkat API."""
	cities = get_us_cities()
	params = load_params("meerkat/web_service/config/web_service.json")
	hyperparams = load_hyperparameters(params)
	meerkat = WebConsumer(params, hyperparams, cities)
	#This thread pool can deal with 'blocking functions' like meerkat.classify
	thread_pool = concurrent.futures.ThreadPoolExecutor(14)

	# pylint: disable=bad-continuation
	with open("meerkat/web_service/schema_input.json") as data_file:
		schema_input = json.load(data_file)

	with open("meerkat/web_service/example_input.json") as data_file:
		example_input = json.load(data_file)

	with open("meerkat/web_service/schema_output.json") as data_file:
		schema_output = json.load(data_file)

	with open("meerkat/web_service/example_output.json") as data_file:
		example_output = json.load(data_file)

	with open("meerkat/web_service/schema_debug_output.json") as data_file:
		schema_debug_output = json.load(data_file)

	with open("meerkat/web_service/example_debug_output.json") as data_file:
		example_debug_output = json.load(data_file)

	@schema.validate(
		input_schema=schema_input,
		input_example=example_input,
		output_schema=schema_output,
		output_example=example_output,
		debug_output_schema=schema_debug_output,
		debug_output_example=example_debug_output
	)

	@gen.coroutine
	def post(self):
		"""Handle post requests asynchonously"""
		data = json.loads(self.request.body.decode())
		# Identify Metadata with Meerkat
		# Futures, threadpools, generator coroutines, and functions as arguments
		# allow us to submit the normally 'blocking function' meerkat.classify
		# and its data to a ThreadPoolExecutor, where one of 14 threads will execute
		# the function to its completion.  However, the 'Future' class encapsulates
		# the execution, which we can return even before the Executor reaches a 'done'
		# state.
		results = yield self.thread_pool.submit(self.meerkat.classify, data)
		#results = self.meerkat.classify(data)
		return results


	def get(self):
		"""Handle get requests"""
		return None

#Print a warning to not execute this file as a module
if __name__ == "__main__":
	print("This module is a Class; it should not be run from the console.")

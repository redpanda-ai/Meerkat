import json

from pprint import pprint
from tornado_json.requesthandlers import APIHandler

from meerkat.web_service.web_consumer import Web_Consumer
from meerkat.web_service import schema
from meerkat.various_tools import load_params, get_us_cities, load_hyperparameters

class Meerkat_API(APIHandler):

	cities = get_us_cities()
	params = load_params("config/web_service.json")
	hyperparams = load_hyperparameters(params)
	meerkat = Web_Consumer(params, hyperparams, cities)

	with open("meerkat/web_service/schema_input.json") as data_file:    
		schema_input = json.load(data_file)

	with open("meerkat/web_service/example_input.json") as data_file:    
		example_input = json.load(data_file)

	with open("meerkat/web_service/schema_output.json") as data_file:    
		schema_output = json.load(data_file)

	with open("meerkat/web_service/example_output.json") as data_file:    
		example_output = json.load(data_file)

	@schema.validate(
		input_schema = schema_input,
		input_example = example_input,
		output_schema = schema_output,
		output_example = example_output
	)

	def post(self):
		"""Handle post requests"""

		data = json.loads(self.request.body.decode())
		
		# Identify Metadata with Meerkat
		results = self.meerkat.classify(data)

		return results

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	print("This module is a Class; it should not be run from the console.")
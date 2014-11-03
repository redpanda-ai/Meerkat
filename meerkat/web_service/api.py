import json

from pprint import pprint
from tornado_json.requesthandlers import APIHandler
from tornado_json import schema

from meerkat.web_service.web_consumer import Web_Consumer
from meerkat.various_tools import load_params, get_us_cities

class Meerkat_API(APIHandler):

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
		output_schema = {},
		output_example = {}
		#output_schema = schema_output,
		#output_example = example_output
	)

	def post(self):
		"""Handle post requests"""

		data = json.loads(self.request.body.decode())
		cities = get_us_cities()
		params = load_params("config/web_service.json")
		hyperparams = load_hyperparameters(params)
		classifier = Web_Consumer(params, data["transaction_list"], hyperparams, cities)

		pprint(data)

		return {

		}
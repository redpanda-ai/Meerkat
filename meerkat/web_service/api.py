import json

from tornado_json.requesthandlers import APIHandler
from tornado_json import schema

class Meerkat(APIHandler):

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

		return {

		}
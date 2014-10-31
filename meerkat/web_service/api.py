import json

from tornado_json.requesthandlers import APIHandler
from tornado_json import schema

class Meerkat(APIHandler):

	@schema.validate(
		input_schema = json.loads("schema_input.json"),
		input_example = json.loads("example_input.json"),
		output_schema = json.loads("schema_output.json"),
		output_example = json.loads("example_output.json")
	)

	def post(self):
	"""Handle post requests"""


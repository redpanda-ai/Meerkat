from tornado import gen
from tornado_json.requesthandlers import APIHandler
from tornado_json import schema

class Meerkat(APIHandler):

	@schema.validate(

	)

	def post(self):
	"""Handle post requests"""
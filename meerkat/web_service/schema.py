"""Process and validate the schema"""
import json
import jsonschema

from functools import wraps
from tornado import gen
from tornado.concurrent import Future

from tornado_json.utils import container

def service_list_validation(services_list):
	if services_list != []:
		services_set = set(services_list)
		combos = [{"cnn_merchant"},
			{"cnn_merchant", "bloom_filter"},
			{"cnn_merchant", "bloom_filter", "search"},
			{"cnn_subtype"},
			{"cnn_merchant", "cnn_subtype"},
			{"cnn_merchant", "cnn_subtype", "bloom_filter"}
			]
		if services_set not in combos:
			raise jsonschema.ValidationError(
				"Invalid services combination. " +\
				"Possible services combinations are {0}.".format(combos)
				)

def validate(input_schema=None, output_schema=None,\
	input_example=None, output_example=None,\
	debug_output_example=None, debug_output_schema=None):
	"""validate schema"""

	@container
	def _validate(rh_method):
		"""Decorator for RequestHandler schema validation

		This decorator:

			- Validates request body against input schema of the method
			- Calls the ``rh_method`` and gets output from it
			- Validates output against output schema of the method
			- Calls ``JSendMixin.success`` to write the validated output

		:type  rh_method: function
		:param rh_method: The RequestHandler method to be decorated
		:returns: The decorated method
		:raises ValidationError: If input is invalid as per the schema
			or malformed
		:raises TypeError: If the output is invalid as per the schema
			or malformed
		"""
		@wraps(rh_method)
		@gen.coroutine
		def _wrapper(self, *args, **kwargs):
			"""Process ``None`` schema"""
			# In case the specified input_schema is ``None``, we
			#   don't json.loads the input, but just set it to ``None``
			#   instead.
			if input_schema is not None:
				# Attempt to json.loads the input
				try:
					# TODO: Assuming UTF-8 encoding for all requests,
					#   find a nice way of determining this from charset
					#   in headers if provided
					encoding = "UTF-8"
					input_ = json.loads(self.request.body.decode(encoding))
				except ValueError as _:
					raise jsonschema.ValidationError(
						"Input is malformed; could not decode JSON object."
					)
				# Validate the received input
				jsonschema.validate(
					input_,
					input_schema
				)
				service_list_validation(input_.get("services_list",[]))
			else:
				input_ = None

			# A json.loads'd version of self.request["body"] is now available
			#   as self.body
			setattr(self, "body", input_)
			# Call the requesthandler method
			output = rh_method(self, *args, **kwargs)
			# If the rh_method returned a Future a la `raise Return(value)`
			#   we grab the output.
			if isinstance(output, Future):
				output = yield output

			if "debug" not in input_ or input_["debug"] == False:
				if output_schema is not None:
					# We wrap output in an object before validating in case
					#  output is a string (and ergo not a validatable JSON object)
					try:
						jsonschema.validate(
							output,
							output_schema
						)
					except jsonschema.ValidationError as err:
						# We essentially re-raise this as a TypeError because
						#  we don't want this error data passed back to the client
						#  because it's a fault on our end. The client should
						#  only see a 500 - Internal Server Error.
						raise TypeError(str(err))
			elif input_["debug"] == True:
				if debug_output_schema is not None:
					# We wrap output in an object before validating in case
					#  output is a string (and ergo not a validatable JSON object)
					try:
						jsonschema.validate(
							output,
							debug_output_schema
						)
					except jsonschema.ValidationError as err:
						# We essentially re-raise this as a TypeError because
						#  we don't want this error data passed back to the client
						#  because it's a fault on our end. The client should
						#  only see a 500 - Internal Server Error.
						raise TypeError(str(err))

			# If no ValidationError has been raised up until here, we write
			#  back output
			self.success(output)

		setattr(_wrapper, "input_schema", input_schema)
		setattr(_wrapper, "output_schema", output_schema)
		setattr(_wrapper, "input_example", input_example)
		setattr(_wrapper, "output_example", output_example)
		setattr(_wrapper, "debug_output_example", debug_output_example)
		setattr(_wrapper, "debug_output_schema", debug_output_schema)


		return _wrapper
	return _validate

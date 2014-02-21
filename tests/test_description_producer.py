'''Unit tests for longtail.description_producer'''

from longtail import description_producer
from longtail.custom_exceptions import InvalidArguments, Misconfiguration
import unittest, queue, sys, socket, os, json

class TokenizeDescriptionTests(unittest.TestCase):

	"""Our UnitTest class."""

	config = """
		{
			"concurrency" : 1,
			"input" : {
				"parameter_key" : "config/keys/made_up_key_name.json",
				"filename" : "data/input/100_bank_transaction_descriptions.csv",
				"encoding" : "utf-8"
			},
			"logging" : {
				"level" : "warning",
				"path" : "logs/foo.log",
				"formatter" : "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
				"console" : true
			},
			"output" : {
				"results" : {
					"fields" : ["BUSINESSSTANDARDNAME", "HOUSE", "PREDIR", "PERSISTENTRECORDID"],
					"size" : 10
				},
				"file" : {
					"format" : "csv",
					"path" : "data/input/unittestDeletable.csv"
				}
			},
			"elasticsearch" : {
				"cluster_nodes" : ["brainstorm0:9200", "brainstorm1:9200", "brainstorm2:9200"
				, "brainstorm3:9200", "brainstorm4:9200", "brainstorm5:9200", "brainstorm6:9200"
				, "brainstorm7:9200", "brainstorm8:9200", "brainstorm9:9200", "brainstorma:9200"
				, "brainstormb:9200"],
				"index" : "new_index",
				"type" : "new_type"
			}
		}"""


	def setUp(self):
		self.params = json.loads(self.config)
		self.desc_queue, self.result_queue = queue.Queue(), queue.Queue()
		for arg in sys.argv[1:]:
			sys.argv.remove(arg)

	def test_usage(self):
		"""The point of this function is to print usage information to the user"""
		result = description_producer.usage()
		self.assertEqual("Usage:\n\t<path_to_json_format_config_file>", result)

	def test_get_desc_queue_returns_queue(self):
		"""Ensure returns an instance of Queue"""
		my_queue, non_physical = description_producer.get_desc_queue(self.params)
		self.assertTrue(isinstance(my_queue, queue.Queue))

	def test_get_desc_queue_is_not_empty(self):
		"""Ensure queue is not empty"""
		my_queue, non_physical = description_producer.get_desc_queue(self.params)
		self.assertFalse(my_queue.empty())

	def test_initialize_no_file_name(self):
		"""Config file not provided"""
		self.assertRaises(InvalidArguments, description_producer.initialize)

	def test_initialize_file_does_not_exist(self):
		"""Config file doesn't exist"""
		sys.argv.append("data/somethingThatWontExist.csv")
		self.assertRaises(SystemExit, description_producer.initialize)

	def test_initialize_too_many_arguments(self):
		"""Too Many Options"""
		sys.argv.append("data/somethingThatWontExist.csv")
		sys.argv.append("argument")
		self.assertRaises(InvalidArguments, description_producer.initialize)

	def test_tokenize(self):
		"""The point of this function is to start a number of
		consumers as well as a starting queue and a result queue.
		At the end a call to write_output_to_file should be made"""

	def test_write_output_to_file_writes_file(self):
		"""Ensure actually writes a file"""
		self.result_queue.put({"PERSISTENTRECORDID":"123456789"})
		description_producer.write_output_to_file(self.params, self.result_queue)
		self.assertTrue(os.path.isfile("data/input/unittestDeletable.csv"))
		os.remove("data/input/unittestDeletable.csv")

	def test_write_output_to_file_empties_queue(self):
		"""Ensure queue is empty at end"""
		self.result_queue.put({"PERSISTENTRECORDID":"123456789"})
		description_producer.write_output_to_file(self.params, self.result_queue)
		self.assertTrue(self.result_queue.empty())
		os.remove("data/input/unittestDeletable.csv")

	def test_validate_logging(self):
		"""Ensure 'logging' key is in configuration"""
		del self.params["logging"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_logging_path(self):
		"""Ensure 'logging.path' key is in configuration"""
		del self.params["logging"]["path"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_elasticsearch_index(self):
		"""Ensure 'elasticsearch.index' key is in configuration"""
		del self.params["elasticsearch"]['index']
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_elasticsearch_type(self):
		"""Ensure 'elasticsearch.type' key is in configuration"""
		del self.params["elasticsearch"]['type']
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_elasticsearch(self):
		"""Ensure 'elasticsearch' key is in configuration"""
		del self.params["elasticsearch"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_empty_config(self):
		"""Ensure configuration is not empty"""
		self.params = {}
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_missing_concurrency(self):
		"""Ensure 'concurrency' key is in configuration"""
		del self.params["concurrency"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_positive_concurrency(self):
		"""Ensure 'concurrency' value is a positive integer"""
		self.params["concurrency"] = 0
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_input_key(self):
		"""Ensure 'input' key is in configuration"""
		del self.params["input"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_input_file(self):
		"""Ensure input file is provided"""
		del self.params["input"]["filename"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_encoding(self):
		"""Ensure encoding key is in configuration"""
		del self.params["input"]["encoding"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_parameter_key_default(self):
		"""Ensure parameter key defaults to default.json"""
		del self.params["input"]["parameter_key"]
		description_producer.validate_params(self.params)
		self.assertEqual(self.params["input"]["parameter_key"], "config/keys/default.json")

	def test_false_key_throws_error(self):
		"""Ensure not existent key throws error"""
		self.assertRaises(SystemExit, description_producer.load_parameter_key, self.params)				

if __name__ == '__main__':
	unittest.main(argv=[sys.argv[0]])

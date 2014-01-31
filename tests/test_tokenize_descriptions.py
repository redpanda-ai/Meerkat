'''Unit tests for longtail.tokenize_descriptions'''

from longtail import tokenize_descriptions
from longtail.custom_exceptions import InvalidArguments
import unittest, queue, sys

class TokenizeDescriptionTests(unittest.TestCase):
	
	"""Our UnitTest class."""

	def setUp(self):
		self.params = {}

	def test_usage(self):
		"""The point of this function is to print usage information to the user"""
		result = tokenize_descriptions.usage()
		self.assertEqual("Usage:\n\t<quoted_transaction_description_string>", result)

	def test_get_desc_queue(self):
		"""The point of this function is to return a non zero desc queue
		containing a list of descriptions as taken from an
		input file provided in params"""

		self.params["input"] = {}
		self.params["input"]["filename"] = "data/100_bank_transaction_descriptions.csv"
		self.params["input"]["encoding"] = "utf-8"
		my_queue = tokenize_descriptions.get_desc_queue(self.params)	
		self.assertTrue(isinstance(my_queue, queue.Queue))
		self.assertFalse(my_queue.empty())		

	def test_get_online_cluster_nodes(self):
		"""The point of this function is to a return a list of online 
		nodes out of a list of possible nodes provided"""

		node_list = ["brainstorm0:9200", "brainstorm1:9200", "brainstorm2:9200"]
		self.params["elasticsearch"] = {}
		self.params["elasticsearch"]["index"] = "new_index"
		self.params["elasticsearch"]["type"] = "new_type"
		self.params["elasticsearch"]["cluster_nodes"] = node_list
		online_nodes = tokenize_descriptions.get_online_cluster_nodes(self.params)

		self.assertNotEqual(len(online_nodes), 0)

	def test_initialize(self):
		"""The point of this function is to return a set of params"""

		# Filename not given
		self.assertRaises(InvalidArguments, tokenize_descriptions.initialize)
		
		# File doesn't exist
		sys.argv.append("data/somethingThatWontExist.csv")
		self.assertRaises(SystemExit, tokenize_descriptions.initialize)

		# Too Many Options
		sys.argv.append("argument")
		self.assertRaises(InvalidArguments, tokenize_descriptions.initialize)

		# Reset
		sys.argv.remove("argument")
		sys.argv.remove("data/somethingThatWontExist.csv")
	
	def test_tokenize(self):
		"""The point of this function is to start a number of 
		consumers as well as a starting queue and a result queue.
		At the end a call to write_output_to_file should be made"""

	def test_write_output_to_file(self):
		"""The point of this function is to write
		to a file. Both CSV and JSON should work"""

	
if __name__ == '__main__':
	unittest.main(argv=[sys.argv[0]])

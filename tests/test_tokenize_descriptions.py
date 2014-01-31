'''Unit tests for longtail.tokenize_descriptions'''

from longtail import tokenize_descriptions
import unittest, queue

class TokenizeDescriptionTests(unittest.TestCase):
	
	"""Our UnitTest class."""

	def setUp(self):
		self.params = {}

	def test_usage(self):
		"""usage test"""
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
		nodes out of a list of possible nodes provided. Should 
		return at least one node from the provided list"""

	def test_initialize(self):
		"""The point of this function is to return a set of params"""
	
	def test_tokenize(self):
		"""The point of this function is to start a number of 
		consumers as well as a starting queue and a result queue.
		At the end a call to write_output_to_file should be made"""

	def test_write_output_to_file(self):
		"""The point of this function is to write to a file. Both CSV and JSON should work"""

	
if __name__ == '__main__':
	unittest.main()

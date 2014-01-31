'''Unit tests for longtail.tokenize_descriptions'''

from longtail import tokenize_descriptions
from longtail.custom_exceptions import InvalidArguments
import unittest, queue, sys, socket, os, json

class TokenizeDescriptionTests(unittest.TestCase):
	
	"""Our UnitTest class."""

	config = """
		{
			"concurrency" : 1,
			"input" : {
				"filename" : "data/100_bank_transaction_descriptions.csv",
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
					"path" : "data/unittestDeletable.csv"
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

	def test_usage(self):
		"""The point of this function is to print usage information to the user"""
		result = tokenize_descriptions.usage()
		self.assertEqual("Usage:\n\t<quoted_transaction_description_string>", result)

	def test_get_desc_queue_returns_queue(self):
		"""Ensure returns an instance of Queue"""
		my_queue = tokenize_descriptions.get_desc_queue(self.params)	
		self.assertTrue(isinstance(my_queue, queue.Queue))

	def test_get_desc_queue_is_not_empty(self):
		"""Ensure queue is not empty"""
		my_queue = tokenize_descriptions.get_desc_queue(self.params)
		self.assertFalse(my_queue.empty())			

	def test_initialize_no_file_name(self):
		"""Config file not provided"""
		self.assertRaises(InvalidArguments, tokenize_descriptions.initialize)
		
	def test_initialize_file_does_not_exist(self):	
		"""Config file doesn't exist"""
		sys.argv.append("data/somethingThatWontExist.csv")
		self.assertRaises(SystemExit, tokenize_descriptions.initialize)
		sys.argv.remove("data/somethingThatWontExist.csv")

	def test_initialize_too_many_arguments(self):
		"""Too Many Options"""
		sys.argv.append("data/somethingThatWontExist.csv")
		sys.argv.append("argument")
		self.assertRaises(InvalidArguments, tokenize_descriptions.initialize)
		sys.argv.remove("argument")
		sys.argv.remove("data/somethingThatWontExist.csv")	
	
	def test_tokenize(self):
		"""The point of this function is to start a number of 
		consumers as well as a starting queue and a result queue.
		At the end a call to write_output_to_file should be made"""

	def test_write_output_to_file_writes_file(self):
		"""Ensure actually writes a file"""
		self.result_queue.put({"PERSISTENTRECORDID":"123456789"})
		tokenize_descriptions.write_output_to_file(self.params, self.result_queue)
		self.assertTrue(os.path.isfile("data/unittestDeletable.csv"))
		os.remove("data/unittestDeletable.csv")

	def test_write_output_to_file_empties_queue(self):
		"""Ensure queue is empty at end"""
		self.result_queue.put({"PERSISTENTRECORDID":"123456789"})
		tokenize_descriptions.write_output_to_file(self.params, self.result_queue)
		self.assertTrue(self.result_queue.empty())
		os.remove("data/unittestDeletable.csv")

	def test_get_online_cluster_nodes_not_empty(self):
		"""Ensure returned node list is not empty"""
		online_nodes = tokenize_descriptions.get_online_cluster_nodes(self.params)
		self.assertNotEqual(len(online_nodes), 0)
		
	def test_get_online_cluster_nodes_are_online(self):
		"""Ensure returned nodes are actually online"""
		online_nodes = tokenize_descriptions.get_online_cluster_nodes(self.params)
		for node in online_nodes:
			node = node.split(":")[0]
			try:
				socket.gethostbyaddr(node)
				self.assertTrue(True)
			except socket.gaierror:
				self.assertTrue(False)				

if __name__ == '__main__':
	unittest.main(argv=[sys.argv[0]])

'''Unit test for longtail.description_consumer'''

import queue, json
from longtail.description_consumer import DescriptionConsumer
from longtail.custom_exceptions import Misconfiguration
import unittest

class DescriptionConsumerTests(unittest.TestCase):
	"""Our UnitTest class."""

	config = """
{
	"concurrency" : 1,
	"input" : {
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
			"size" : 1
		},
		"file" : {
			"format" : "json",
			"path" : "data/output/longtailLabeled.json"
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

	clean_my_meta = """
{
	"unigram_tokens" : [],
	"tokens" : [],
	"metrics" : {
		"query_count" : 0,
		"cache_count" : 0
	}
}"""

	def setUp(self):
		"""Basic Fixture for all tests."""
		self.params = json.loads(self.config)
		self.desc_queue, self.result_queue = queue.Queue, queue.Queue

	def test_validate_elasticsearch(self):
		"""Ensure 'elasticsearch' key is in configuration"""
		del self.params["elasticsearch"]
		args = [0, self.params, self.desc_queue, self.result_queue]
		self.assertRaises(Misconfiguration, DescriptionConsumer, *args)

	def test_validate_empty_config(self):
		"""Ensure configuration is not empty"""
		self.params = {}
		args = [0, self.params, self.desc_queue, self.result_queue]
		self.assertRaises(Misconfiguration, DescriptionConsumer, *args)

	def test_validate_missing_concurrency(self):
		"""Ensure 'concurrency' key is in configuration"""
		del self.params["concurrency"]
		args = [0, self.params, self.desc_queue, self.result_queue]
		self.assertRaises(Misconfiguration, DescriptionConsumer, *args)

	def test_validate_positive_concurrency(self):
		"""Ensure 'concurrency' value is a positive integer"""
		self.params["concurrency"] = 0
		args = [0, self.params, self.desc_queue, self.result_queue]
		self.assertRaises(Misconfiguration, DescriptionConsumer, *args)

	def test_validate_input_key(self):
		"""Ensure 'input' key is in configuration"""
		del self.params["input"]
		args = [0, self.params, self.desc_queue, self.result_queue]
		self.assertRaises(Misconfiguration, DescriptionConsumer, *args)

	def test_display_z_score_single_score(self):
		"""Ensure that list containing one score, returns None for z_score"""
		scores = [0]
		my_consumer = DescriptionConsumer(0, self.params, self.desc_queue
		, self.result_queue)
		result = my_consumer._DescriptionConsumer__display_z_score_delta(scores)
		self.assertEqual(result,None)

	def test_display_z_score_delta(self):
		"""Ensure that list containing [1,2,3], returns -1.225 for z_score"""
		scores = [1, 2, 3]
		my_consumer = DescriptionConsumer(0, self.params, self.desc_queue
		, self.result_queue)
		result = my_consumer._DescriptionConsumer__display_z_score_delta(scores)
		self.assertEqual(result,-1.225)

	def test_validate_logging(self):
		"""Ensure 'logging' key is in configuration"""
		del self.params["logging"]
		args = [0, self.params, self.desc_queue, self.result_queue]
		self.assertRaises(Misconfiguration, DescriptionConsumer, *args)

	def test_validate_logging_path(self):
		"""Ensure 'logging.path' key is in configuration"""
		del self.params["logging"]["path"]
		args = [0, self.params, self.desc_queue, self.result_queue]
		self.assertRaises(Misconfiguration, DescriptionConsumer, *args)

	def test_validate_elasticsearch_index(self):
		"""Ensure 'elasticsearch.index' key is in configuration"""
		del self.params["elasticsearch"]['index']
		args = [0, self.params, self.desc_queue, self.result_queue]
		self.assertRaises(Misconfiguration, DescriptionConsumer, *args)

	def test_validate_elasticsearch_type(self):
		"""Ensure 'elasticsearch.type' key is in configuration"""
		del self.params["elasticsearch"]['type']
		args = [0, self.params, self.desc_queue, self.result_queue]
		self.assertRaises(Misconfiguration, DescriptionConsumer, *args)

	def test_reset_my_meta_recursive(self):
		"""Ensure that the 'recursive' memeber is reset to 'false'"""
		my_consumer = DescriptionConsumer(0, self.params, self.desc_queue
		, self.result_queue)
		my_consumer.recursive = True
		my_consumer._DescriptionConsumer__reset_my_meta()
		self.assertEqual(my_consumer.recursive,False)

	def test_reset_my_meta_n_gram_tokens(self):
		"""Ensure that the 'recursive' memeber is reset to 'false'"""
		my_consumer = DescriptionConsumer(0, self.params, self.desc_queue
		, self.result_queue)
		my_consumer.n_gram_tokens = {"not" : "empty"}
		my_consumer._DescriptionConsumer__reset_my_meta()
		self.assertEqual(my_consumer.n_gram_tokens, {})

	def test_reset_my_meta_n_gram_tokens(self):
		"""Ensure that the 'my_meta' memeber is reset"""
		my_consumer = DescriptionConsumer(0, self.params, self.desc_queue
		, self.result_queue)
		my_consumer.my_meta = {"dirty" : "my_meta"}
		my_consumer._DescriptionConsumer__reset_my_meta()
		#expected = = json.loads(self.clean_my_meta)
		self.assertEqual(my_consumer.my_meta, json.loads(self.clean_my_meta))

if __name__ == '__main__':
	unittest.main()


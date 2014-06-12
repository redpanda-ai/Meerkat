"""Unit tests for meerkat.description_consumer"""

import collections
import json
import numpy as np
import queue
import unittest
from meerkat.description_consumer import DescriptionConsumer
from meerkat.custom_exceptions import Misconfiguration

class DescriptionConsumerTests(unittest.TestCase):
	"""Our UnitTest class."""

	clean_my_meta = '{"metrics" : {"query_count" : 0}}'

	config = """{
		"concurrency" : 1,
		"input" : {
			"filename" : "data/input/100_bank_transaction_descriptions.csv",
			"encoding" : "utf-8"
		},
		"logging" : {
			"level" : "warning", "path" : "logs/foo.log", "console" : false,
			"formatter" : "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
		},
		"output" : {
			"results" : {
				"fields" : ["name", "factual_id", "pin.location", "locality", "region"],
				"size" : 1
			},
			"file" : {
				"format" : "json", "path" : "data/output/meerkatLabeled.json"
			}
		},
		"elasticsearch" : {
			"cluster_nodes" : [
			    "s01:9200",
			    "s02:9200",
			    "s03:9200",
			    "s04:9200",
			    "s05:9200",
			    "s06:9200",
			    "s07:9200",
			    "s08:9200",
			    "s09:9200",
			    "s10:9200",
			    "s11:9200",
			    "s12:9200",
			    "s13:9200",
			    "s14:9200",
			    "s15:9200",
			    "s16:9200",
			    "s17:9200",
			    "s18:9200"
	    	],
			"index" : "factual_index", "type" : "factual_type",
			"boost_labels" : [ "standard_fields", "composite.address" ],
			"boost_vectors" : {
				"factual_id" :        [ 0.0, 1.0 ],
				"name" :              [ 1.0, 0.0 ],
				"address" :           [ 0.0, 1.0 ]
			}
		},
		"search_cache" : {}
	}"""

	search_results = """{
		"hits": {
			"hits": [
				{"_score": 3, "_type": "new_type", "_index": "factual_index",
				"_id": "4", "fields": {"factual_id": "4", "name" : "name1"}},
				{"_score": 2, "_type": "new_type", "_index": "factual_index",
				"_id": "6", "fields": {"factual_id": "6", "name" : "name1"}},
				{"_score": 1, "_type": "new_type", "_index": "factual_index",
				"_id": "9", "fields": {"factual_id": "9", "name" : "name1"}}
			],
			"total": 3,
			"max_score": 3
		},
		"_shards": {
			"successful": 12, 
			"failed": 0, 
			"total": 12
		}, 
		"took": 100, 
		"timed_out": false
	}"""

	input_json = """{
		"query": {
			"bool": {
				"should": [
					{"match": {"_all": {"type": "phrase", "query": "SUNNYVALE"}}}
				], 
				"minimum_number_should_match": 1
			}
		},
		"from": 0, 
		"size": 0
	}"""

	list_compare = lambda self, x, y: collections.Counter(x) == collections.Counter(y)
	my_consumer, hyperparameters = None, ' {"es_result_size":"20"} '

	def setUp(self):
		"""Basic Fixture for all tests."""
		self.hyperparameters = json.loads(self.hyperparameters)
		self.params = json.loads(self.config)
		self.desc_queue, self.result_queue = queue.Queue(), queue.Queue()
		self.my_consumer = DescriptionConsumer(0, self.params, self.desc_queue
			, self.result_queue, self.hyperparameters)

	def test_display_z_score_single_score(self):
		"""Ensure that list containing one score, returns None for z_score"""
		scores = [0]
		result = self.my_consumer._DescriptionConsumer__generate_z_score_delta(scores)
		self.assertEqual(result, 0)

	def test_generate_z_score_delta(self):
		"""Ensure that list containing [3, 2, 1], returns 1.225 for z_score"""
		scores = [3, 2, 1]
		result = self.my_consumer._DescriptionConsumer__generate_z_score_delta(scores)
		self.assertEqual(result, 1.225)

	def test_display_search_results_normal_use(self):
		"""Ensure that display_search_results method completes """
		search_results = json.loads(self.search_results)
		transaction = {"DESCRIPTION" : "Ham Sandwich"}
		result = self.my_consumer._DescriptionConsumer__display_search_results(search_results, transaction)
		self.assertEqual(result,True)

	def test_output_to_result_queue(self):
		"""Ensure that we can output to the result queue"""
		search_results = json.loads(self.search_results)
		self.my_consumer._DescriptionConsumer__output_to_result_queue(search_results)
		self.assertEqual(False,self.my_consumer.result_queue.empty())

	def test_reset_my_meta_reset_my_meta(self):
		"""Ensure that the 'my_meta' member is reset"""
		self.my_consumer.my_meta = {"dirty" : "my_meta"}
		self.my_consumer._DescriptionConsumer__reset_my_meta()
		self.assertEqual(self.my_consumer.my_meta, json.loads(self.clean_my_meta))

	def test_search_index_normal_use(self):
		"""Ensure that __search_index finds a common result."""
		input_as_object = json.loads(self.input_json)
		result = self.my_consumer._DescriptionConsumer__search_index(input_as_object)
		self.assertGreater(result["hits"]["total"], -1)

	def test_build_boost_vectors_boost_row_labels(self):
		"""Ensure that __build_boost_vectors correctly builds a sorted list of
		boost_row_labels."""
		result, _ = self.my_consumer._DescriptionConsumer__build_boost_vectors()
		expect = ["address", "factual_id", "name"]
		self.assertEqual(self.list_compare(result,expect), True)

	def test_build_boost_vectors_boost_column_vectors(self):
		"""Ensure that __build_boost_vectors correctly builds a dictionary of
		boost_column_vectors."""
		_, result = self.my_consumer._DescriptionConsumer__build_boost_vectors()
		expect = {
			"composite.address": np.array([1, 1, 0]),
			"standard_fields": np.array([0, 0, 1]),
		}
		for key in result.keys():
			r, e = result[key], expect[key]
			self.assertTrue(np.allclose(r, e, rtol=1e-05, atol=1e-08))

	def test_get_boosted_fields_normal_use(self):
		"""Test that __get_boosted_fields returns the expected result"""
		self.my_consumer.boost_column_vectors = { "vector_1" : [0.0, 2.0, 1.0] }
		self.my_consumer.boost_row_labels = ["A", "B", "C"]
		expect = ["B^2.0", "C^1.0"]
		result = self.my_consumer._DescriptionConsumer__get_boosted_fields("vector_1")
		self.assertEqual(self.list_compare(result,expect), True)

if __name__ == '__main__':
	unittest.main()

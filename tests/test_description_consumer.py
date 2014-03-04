'''Unit tests for longtail.description_consumer'''

import collections
import json
import numpy as np
import queue
import unittest
from longtail.description_consumer import DescriptionConsumer
from longtail.custom_exceptions import Misconfiguration

class DescriptionConsumerTests(unittest.TestCase):
	"""Our UnitTest class."""

	clean_my_meta = """
{
	"unigram_tokens" : [], "tokens" : [],
	"metrics" : { "query_count" : 0, "cache_count" : 0 }
}"""

	config = """
{
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
			"fields" : ["BUSINESSSTANDARDNAME", "HOUSE", "PREDIR", "PERSISTENTRECORDID"],
			"size" : 1
		},
		"file" : {
			"format" : "json", "path" : "data/output/longtailLabeled.json"
		}
	},
	"elasticsearch" : {
		"cluster_nodes" : ["brainstorm0:9200", "brainstorm1:9200", "brainstorm2:9200"
		, "brainstorm3:9200", "brainstorm4:9200", "brainstorm5:9200", "brainstorm6:9200"
		, "brainstorm7:9200", "brainstorm8:9200", "brainstorm9:9200", "brainstorma:9200"
		, "brainstormb:9200"],
		"index" : "new_index", "type" : "new_type",
		"subqueries" : {
			"largest_matching_string": {
				"field_boosts" : "standard_fields",
				"query_type" : "qs_query"
			},
			"find_addresses": {
				"field_boosts" : "composite.address",
				"query_type" : "multi_match_query"
			}
		},
		"boost_labels" : [ "standard_fields", "composite.address" ],
		"boost_vectors" : {
			"factual_id" :        [ 0.0, 1.0 ],
			"name" :              [ 1.0, 0.0 ],
			"address" :           [ 0.0, 1.0 ]
		}
	},
	"search_cache" : {}
}"""

	search_results = """
{
	"hits": {
		"hits": [
			{"_score": 3, "_type": "new_type", "_index": "new_index",
			"_id": "4", "fields": {"PERSISTENTRECORDID": "4"}},
			{"_score": 2, "_type": "new_type", "_index": "new_index",
			"_id": "6", "fields": {"PERSISTENTRECORDID": "6"}},
			{"_score": 1, "_type": "new_type", "_index": "new_index",
			"_id": "9", "fields": {"PERSISTENTRECORDID": "9"}}
		],
	"total": 3,
	"max_score": 3
	},
	"_shards": {"successful": 12, "failed": 0, "total": 12}, "took": 100, "timed_out": false
}"""

	input_json = """
{
	"query": {
		"bool": {
			"should": [
				{"match": {"_all": {"type": "phrase", "query": "SUNNYVALE"}}}
			], 
			"minimum_number_should_match": 1
		}
	},
	"from": 0, "size": 0
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

	def test_begin_parse_no_string(self):
		"""Ensure that __begin_parse fails gracefully with None strings"""
		result = self.my_consumer._DescriptionConsumer__begin_parse()
		self.assertEqual(result,False)

	def test_begin_parse_empty_string(self):
		"""Ensure that __begin_parse fails gracefully with empty strings"""
		result = self.my_consumer._DescriptionConsumer__begin_parse()
		self.assertEqual(result,False)

	def test_break_string_into_substrings_normal_use(self):
		"""Ensure that we can recursively discover all substrings"""
		term, substrings = "ABCD", {}
		result = self.my_consumer._DescriptionConsumer__break_string_into_substrings(term, substrings)
		expect = {2: {'AB': '', 'CD': '', 'BC': ''}, 3: {'BCD': '', 'ABC': ''}, 4: {'ABCD': ''}}
		self.assertEqual(substrings,expect)

	def test_display_z_score_single_score(self):
		"""Ensure that list containing one score, returns None for z_score"""
		scores = [0]
		result = self.my_consumer._DescriptionConsumer__display_z_score_delta(scores)
		self.assertEqual(result,None)

	def test_display_z_score_delta(self):
		"""Ensure that list containing [3, 2, 1], returns 1.225 for z_score"""
		scores = [3, 2, 1]
		result = self.my_consumer._DescriptionConsumer__display_z_score_delta(scores)
		self.assertEqual(result, 1.225)

	def test_display_search_results_normal_use(self):
		"""Ensure that display_search_results method completes """
		search_results = json.loads(self.search_results)
		result = self.my_consumer._DescriptionConsumer__display_search_results(search_results)
		self.assertEqual(result,True)

	def test_get_multi_gram_tokens_normal_use(self):
		"""Ensure that n-grams where n >=1 can be generated"""
		list_of_tokens = ["1", "2", "3", "4"]
		self.my_consumer._DescriptionConsumer__get_multi_gram_tokens(list_of_tokens)
		result = self.my_consumer.multi_gram_tokens
		expect = {1: ['4', '3', '2', '1'], 2: ['3 4', '2 3', '1 2'],
			3: ['2 3 4', '1 2 3'], 4: ['1 2 3 4']}
		self.assertEqual(result,expect)

	def test_output_to_result_queue(self):
		"""Ensure that we can output to the result queue"""
		search_results = json.loads(self.search_results)
		self.my_consumer._DescriptionConsumer__output_to_result_queue(search_results)
		self.assertEqual(False,self.my_consumer.result_queue.empty())

	def test_rebuild_tokens_normal_use(self):
		"""Ensure that __rebuild_tokens works with a standard case"""
		term, substring, pre, post = "BCDEF", "CDE", "B", "F"
		self.my_consumer.my_meta["tokens"] = ["A", "BCDEF", "GH"]
		self.my_consumer._DescriptionConsumer__rebuild_tokens(
			term, substring, pre, post)
		result = self.my_consumer.my_meta["tokens"]
		expect = ["A", "B", "CDE", "F", "GH"]
		self.assertEqual(self.list_compare(result, expect), True)

	def test_reset_my_meta_recursive(self):
		"""Ensure that the 'recursive' member is reset to 'false'"""
		self.my_consumer.recursive = True
		self.my_consumer._DescriptionConsumer__reset_my_meta()
		self.assertEqual(self.my_consumer.recursive,False)

	def test_reset_my_meta_multi_gram_tokens(self):
		"""Ensure that the 'multi_gram_tokens' dict is emptied"""
		self.my_consumer.multi_gram_tokens = {"not" : "empty"}
		self.my_consumer._DescriptionConsumer__reset_my_meta()
		self.assertEqual(self.my_consumer.multi_gram_tokens, {})

	def test_reset_my_meta_reset_my_meta(self):
		"""Ensure that the 'my_meta' member is reset"""
		self.my_consumer.my_meta = {"dirty" : "my_meta"}
		self.my_consumer._DescriptionConsumer__reset_my_meta()
		self.assertEqual(self.my_consumer.my_meta, json.loads(self.clean_my_meta))

	def test_search_index_normal_use(self):
		"""Ensure that __search_index finds a common result."""
		input_as_object = json.loads(self.input_json)
		result = self.my_consumer._DescriptionConsumer__search_index(input_as_object)
		self.assertGreater(result["hits"]["total"],-1)

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

	def test_get_subquery_qs_query(self):
		"""Test that __get_subquery works with a named qs_query subquery."""
		expect = { 'query_string': { 'boost': 1.0, 'fields': ['name^1.0'], 'query': 'some_term'} }
		result = self.my_consumer._DescriptionConsumer__get_subquery("some_term", "largest_matching_string")
		self.assertEqual(expect, result)

	def test_get_subquery_multi_match_query(self):
		"""Test that __get_subquery works with a named multi_match_query subquery."""
		expect = { 'multi_match': { 'fields': ['address^1.0', 'factual_id^1.0'],
			'query': 'some_address', 'type': 'phrase'} }
		result = self.my_consumer._DescriptionConsumer__get_subquery("some_address", "find_addresses")
		self.assertEqual(expect, result)

	def test_get_subquery_named_query_not_found(self):
		"""Test that __get_subquery works with a named multi_match_query subquery."""
		self.assertRaises(Misconfiguration, self.my_consumer._DescriptionConsumer__get_subquery,
			"some_term", "not_in_configuration")

#	def test_find_largest_matching_string(self):
#		"""Ensure that __find_largest_matching_string finds a known string."""

if __name__ == '__main__':
	unittest.main()


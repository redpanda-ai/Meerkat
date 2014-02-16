'''Unit tests for longtail.description_consumer'''

import collections, json, queue, unittest
from longtail.description_consumer import DescriptionConsumer

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
		"level" : "warning", "path" : "logs/foo.log", "console" : true,
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
		"index" : "new_index", "type" : "new_type"
	}
}"""

	search_results = """
{
	"hits": {
		"hits": [
			{"_score": 0.054203592, "_type": "new_type",
			"_id": "40404997", "fields": {"PERSISTENTRECORDID": "40404997"}, "_index": "new_index"},
			{"_score": 0.043917503, "_type": "new_type",
			"_id": "695294798", "fields": {"PERSISTENTRECORDID": "695294798"}, "_index": "new_index"},
			{"_score": 0.04222678, "_type": "new_type",
			"_id": "16718350", "fields": {"PERSISTENTRECORDID": "16718350"}, "_index": "new_index"}
		],
	"total": 3,
	"max_score": 0.054203592
	},
	"_shards": {"successful": 12, "failed": 0, "total": 12}, "took": 100, "timed_out": false}
"""

	list_compare = lambda self, x, y: collections.Counter(x) == collections.Counter(y)
	my_consumer, parameter_key = None, ' {"es_result_size":"20"} '

	def setUp(self):
		"""Basic Fixture for all tests."""
		self.parameter_key = json.loads(self.parameter_key)
		self.params = json.loads(self.config)
		self.desc_queue, self.result_queue = queue.Queue(), queue.Queue()
		self.my_consumer = DescriptionConsumer(0, self.params, self.desc_queue
			, self.result_queue, self.parameter_key)

	def test_begin_parse_no_string(self):
		"""Ensure that __begin_parse fails gracefully with None strings"""
		result = self.my_consumer._DescriptionConsumer__begin_parse()
		self.assertEqual(result,False)

	def test_begin_parse_empty_string(self):
		"""Ensure that __begin_parse fails gracefully with empty strings"""
		result = self.my_consumer._DescriptionConsumer__begin_parse()
		self.assertEqual(result,False)

	def test_display_z_score_single_score(self):
		"""Ensure that list containing one score, returns None for z_score"""
		scores = [0]
		result = self.my_consumer._DescriptionConsumer__display_z_score_delta(scores)
		self.assertEqual(result,None)

	def test_display_z_score_delta(self):
		"""Ensure that list containing [1,2,3], returns -1.225 for z_score"""
		scores = [1, 2, 3]
		result = self.my_consumer._DescriptionConsumer__display_z_score_delta(scores)
		self.assertEqual(result,-1.225)

	def test_get_n_gram_tokens_normal_use(self):
		"""Ensure that n-grams where n >=2 can be generated"""
		list_of_tokens = ["1", "2", "3", "4"]
		self.my_consumer._DescriptionConsumer__get_n_gram_tokens(list_of_tokens)
		result = self.my_consumer.n_gram_tokens
		expect = {2: ['3 4', '2 3', '1 2'], 3: ['2 3 4', '1 2 3'], 4: ['1 2 3 4']}
		self.assertEqual(result,expect)

	def test_output_to_result_queue(self):
		"""Ensure that we can output to the result queue"""
		search_results = json.loads(self.search_results)
		self.my_consumer._DescriptionConsumer__output_to_result_queue(search_results)
		self.assertEqual(False,self.my_consumer.result_queue.empty())

	def test_powerset_normal_use(self):
		"""Ensure that we can recursively discover all substrings"""
		term, substrings = "ABCD", {}
		result = self.my_consumer._DescriptionConsumer__powerset(term, substrings)
		expect = {2: {'AB': '', 'CD': '', 'BC': ''}, 3: {'BCD': '', 'ABC': ''}, 4: {'ABCD': ''}}
		self.assertEqual(substrings,expect)

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

	def test_reset_my_meta_n_gram_tokens(self):
		"""Ensure that the 'n_gram_tokens' dict is emptied"""
		self.my_consumer.n_gram_tokens = {"not" : "empty"}
		self.my_consumer._DescriptionConsumer__reset_my_meta()
		self.assertEqual(self.my_consumer.n_gram_tokens, {})

	def test_reset_my_meta_reset_my_meta(self):
		"""Ensure that the 'my_meta' member is reset"""
		self.my_consumer.my_meta = {"dirty" : "my_meta"}
		self.my_consumer._DescriptionConsumer__reset_my_meta()
		self.assertEqual(self.my_consumer.my_meta, json.loads(self.clean_my_meta))

if __name__ == '__main__':
	unittest.main()


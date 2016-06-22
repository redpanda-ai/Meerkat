"""Unit test for meerkat.elasticsearch.load_index_from_file"""

import argparse
import sys
import unittest

import meerkat.elasticsearch.load_index_from_file as loader

from nose_parameterized import parameterized
from elasticsearch import Elasticsearch

def create_parser():
	"""Creates an argparse parser."""
	parser = argparse.ArgumentParser()
	parser.add_argument("configuration_file")
	return parser

class LoadIndexFromFileTests (unittest.TestCase):
	"""Our UnitTest class."""

	@classmethod
	def setUpClass(cls):
		cls._es = Elasticsearch("172.31.19.192:9200" , sniff_on_start=False,
			sniff_on_connection_fail=True, sniffer_timeout=5, sniff_timeout=5)

	def test_foo(self):
		"""Sample unit test."""
		self.assertTrue(True)


	@parameterized.expand([
		(False, ["meerkat/elasticsearch/config/factual_loader.json"])
	])
	def test_parse_arguments(self, exception_test, arguments):
		"""Simple test to ensure that this function works"""
		if not exception_test:
			results = loader.parse_arguments(arguments)
			expected = create_parser().parse_args(arguments)
			self.assertEqual(results, expected)

	@parameterized.expand([
		(None, "tests/elasticsearch/fixtures/small_doc.tab"),
	])
	def test_load_document_queue(self, exception_test, filename):
		params = {
			"input" : {
				"filename": filename,
				"encoding": "utf-8"
			}
		}
		if not exception_test:
			header, document_queue, document_queue_populated = loader.load_document_queue(params)
			self.assertEqual(header, ['ONE', 'TWO', 'THREE'])
			self.assertEqual(document_queue.qsize(), 2)
			self.assertTrue(document_queue_populated)

	def test_guarantee_index_and_doc_type(self):
		#Data fixture
		params = { "elasticsearch" : {
			"cluster_nodes" : [ "172.31.19.192:9200" ],
			"index" : "dev_index",
			"type" : "dev_type",
			"type_mapping" : {} } }
		#Clean up pre-existing index
		self._es.indices.delete(index=params["elasticsearch"]["index"], ignore=[400, 404])
		#Test
		expected = ("created", "created")
		result = loader.guarantee_index_and_doc_type(params)
		self.assertEqual(expected, result)
		#Clean up pre-existing index
		self._es.indices.delete(index=params["elasticsearch"]["index"], ignore=[400, 404])

if __name__ == '__main__':
	unittest.main()

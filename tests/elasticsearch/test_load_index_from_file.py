"""Unit test for meerkat.elasticsearch.load_index_from_file"""

import argparse
import sys
import unittest

import meerkat.elasticsearch.load_index_from_file as loader

from nose_parameterized import parameterized

def create_parser():
	"""Creates an argparse parser."""
	parser = argparse.ArgumentParser()
	parser.add_argument("configuration_file")
	return parser

class LoadIndexFromFileTests (unittest.TestCase):
	"""Our UnitTest class."""

	@classmethod
	def setUpClass(cls):
		pass

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

if __name__ == '__main__':
	unittest.main()

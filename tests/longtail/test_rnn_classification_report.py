"""Unit test for meerkat/longtail/rnn_classification_report.py"""

import os
import meerkat.longtail.rnn_classification_report as cr
import unittest

from nose_parameterized import parameterized

class RnnClassificationReportTests(unittest.TestCase):
	"""Unit tests for meerkat/longtail/rnn_classification_report.py """

	@parameterized.expand([
		("tests/longtail/fixture/test_get_write_func.csv",
			[{"ground_truth": [], "Predicted": [[1,0], [0,1]], "Description": "Purchase Amazon"}],
			True
		),
		("tests/longtail/fixture/test_get_write_func.csv", [], False)
	])
	def test_get_write_func(self, filename, data, exist):
		config = {"tag_map": {"0": "background", "1": "merchant"},
					"max_tokens": 35
				}
		write_data = cr.get_write_func(filename, config)
		write_data(data)
		self.assertEqual(os.path.isfile(filename), exist)
		if exist:
			os.remove(filename)

	@parameterized.expand([
		({"ground_truth": [], "Predicted": [[1,0], [0,1]], "Description": "Purchase Amazon"},
			"Amazon"),
		({"ground_truth": 123, "Predicted": [[1,0], [1,0], [1,0]], "Description": "payment thank you"},
			"")
	])
	def test_beautify(self, item, expected):
		config = {"tag_map": {"0": "background", "1": "merchant"},
					"max_tokens": 35
				}
		result = cr.beautify(item, config)
		self.assertEqual(result["Predicted"], expected)

	@parameterized.expand([
		(False, ["data", "model", "w2i", "--config", "SNOZ"])
	])
	def test_parse_arguments(self, exception_test, arguments):
		"""Simple test to ensure that this function works"""
		if not exception_test:
			parser = cr.parse_arguments(arguments)
			self.assertEqual(parser.model, "model")
			self.assertEqual(parser.data, "data")
			self.assertEqual(parser.w2i, "w2i")
			self.assertEqual(parser.config, "SNOZ")


if __name__ == '__main__':
	unittest.main()


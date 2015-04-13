"""Unit tests for meerkat.file_producer"""

import unittest, queue, sys, socket, os, json

from meerkat import file_producer
from meerkat.custom_exceptions import InvalidArguments, Misconfiguration
from meerkat.classification.load import select_model

class TokenizeDescriptionTests(unittest.TestCase):

	"""Our UnitTest class."""


	def setUp(self):
		try:
			input_file = open("config/daemon/file.json", encoding='utf-8')
			sys.argv.append("foo")
			sys.argv.append("bar")
			self.params = json.loads(input_file.read())
			input_file.close()
		except:
			print("Cannot find config")
			sys.exit()
		self.desc_queue, self.result_queue = queue.Queue(), queue.Queue()

	def test_usage(self):
		"""The point of this function is to print usage information to the user"""
		result = file_producer.usage()
		self.assertEqual("Usage:\n\t<location_pair_name> <file_name>", result)

if __name__ == '__main__':
	unittest.main(argv=[sys.argv[0]])

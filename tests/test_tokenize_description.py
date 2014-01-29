'''Unit tests for longtail.tokenize_descriptions'''

import longtail.tokenize_descriptions
import unittest

class TokenizeDescriptionTests(unittest.TestCase):
	
	"""Our UnitTest class."""
	def test_usage(self):
		"""usage test"""
		result = longtail.tokenize_descriptions.usage()
		self.assertEqual("Usage:\n\t<quoted_transaction_description_string>", result)

if __name__ == '__main__':
	unittest.main()

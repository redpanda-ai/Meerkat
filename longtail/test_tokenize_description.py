'''Unit test for roman1.py

This program is part of 'Dive Into Python 3', a free Python book for
experienced programmers.  Visit http://diveintopython3.org/ for the
latest version.
'''

import longtail.tokenize_descriptions
import unittest

class TokenizeDescriptionTests(unittest.TestCase):
	"""Our UnitTest class."""
	#We could put stuff here, like variables
	def test_usage(self):
		"""A simple test."""
		result = longtail.tokenize_descriptions.usage()
		self.assertEqual("Usage:\n\t<quoted_transaction_description_string>", result)

if __name__ == '__main__':
	unittest.main()

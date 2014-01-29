'''Unit test for longtail.various_tools'''

import longtail.various_tools
import unittest

class VariousToolsTests(unittest.TestCase):

	"""Our UnitTest class."""
	strings = (("Regular words[]'{}/:", "Regular words"),
                ('Trans with "" quotes', 'Trans with  quotes'))

	numbers = (("[8]6{7}5'3/0-9", "8675309"),
				('333"6"6"6"999', '333666999'))

	def test_string_cleanse(self):
		"""string_cleanse test"""

		for chars, no_chars in self.strings:
			result = longtail.various_tools.string_cleanse(chars)
			self.assertEqual(no_chars, result)

	def test_numeric_cleanse(self):
		"""numeric_cleanse test"""

		for chars, no_chars in self.numbers:
			result = longtail.various_tools.numeric_cleanse(chars)
			self.assertEqual(no_chars, result)

if __name__ == '__main__':
	unittest.main()

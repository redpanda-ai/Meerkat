'''Unit test for meerkat.various_tools'''

import meerkat.various_tools
import numpy as np
import unittest

class VariousToolsTests(unittest.TestCase):

	"""Our UnitTest class."""

	strings = (("Weekend at Bernie's[]{}/:", "Weekend at Bernie's"),
                ('Trans with "" quotes', 'Trans with quotes'))

	numbers = (("[8]6{7}5'3/0-9", "8675309"),
				('333"6"6"6"999', '333666999'))

	original_polygon = [
		[-122.392586, 37.782428], [-122.434139, 37.725378], [-122.462813, 37.725407],
		[-122.48432, 37.742723], [-122.482605, 37.753909], [-122.476587, 37.784143],
		[-122.446137, 37.798541], [-122.419482, 37.807829], [-122.418104, 37.808003],
		[-122.413038, 37.807794], [-122.397797, 37.792259], [-122.392586, 37.782428]]

	doubled_original_polygon = [
		[-122.35015583333332, 37.78895250000001], [-122.43326183333333, 37.6748525],
		[-122.49060983333332, 37.674910499999996], [-122.53362383333332, 37.7095425],
		[-122.53019383333334, 37.7319145], [-122.51815783333332, 37.7923825],
		[-122.45725783333332, 37.8211785], [-122.40394783333333, 37.8397545],
		[-122.40119183333333, 37.8401025], [-122.39105983333333, 37.839684500000004],
		[-122.36057783333332, 37.808614500000004], [-122.35015583333332, 37.78895250000001]]

	def test_string_cleanse(self):
		"""string_cleanse test"""

		for chars, no_chars in self.strings:
			result = meerkat.various_tools.string_cleanse(chars)
			self.assertEqual(no_chars, result)

	def test_numeric_cleanse(self):
		"""numeric_cleanse test"""
		for chars, no_chars in self.numbers:
			result = meerkat.various_tools.numeric_cleanse(chars)
			self.assertEqual(no_chars, result)

	def test_scale_polygon__centroid(self):
		"""scale_polygon test that it correctly calculates the centroid"""
		expect = [[-122.43501617, 37.7759035]]
		result, _, _, _ = meerkat.various_tools.scale_polygon(self.original_polygon, scale = 2.0)
		self.assertTrue(np.allclose(expect, result, rtol=1e-05, atol=1e-08))

	def test_scale_polygon__scaled_list_of_points(self):
		"""scale_polygon test that it converts a list of polygon points to a numpy matrix"""
		expect = self.doubled_original_polygon
		_, result, _, _ = meerkat.various_tools.scale_polygon(self.original_polygon, scale = 2.0)
		self.assertTrue(np.allclose(np.matrix(expect), np.matrix(result), rtol=1e-05, atol=1e-08))

	def test_scale_polygon__original_matrix(self):
		"""scale_polygon test that it converts a list of polygon points to a numpy matrix"""
		expect = np.matrix(self.original_polygon)
		_, _, result, _ = meerkat.various_tools.scale_polygon(self.original_polygon, scale = 2.0)
		self.assertTrue(np.allclose(expect, result, rtol=1e-05, atol=1e-08))

	def test_scale_polygon__scaled_matrix(self):
		"""scale_polygon test that it produces a correctly scaled matrix"""
		expect = np.matrix(self.doubled_original_polygon)
		_, _, _, result = meerkat.various_tools.scale_polygon(self.original_polygon, scale = 2.0)
		self.assertTrue(np.allclose(expect, result, rtol=1e-05, atol=1e-08))

if __name__ == '__main__':
	unittest.main()

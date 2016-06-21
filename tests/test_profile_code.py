"""Unit test for meerkat.profile_code"""

import unittest
import os
import meerkat.profile_code as profile_code
from nose_parameterized import parameterized

class ProfileCodeTest(unittest.TestCase):
	"UnitTest class for profile_code.py"""

	@classmethod
	def tearDownClass(cls):
		try:
			os.remove("auto_load_profiling.csv")
		except OSError:
			pass

	@parameterized.expand([
		([['ab','c','d','efg','meerkat/classification/fake_module.py']], "fake_module"),
		([[2, 'bac','ggg','xyz','Meerkat.classification.fake_module.py']], "unknown_module"),
		([[3, 'a','meerkat/temp.py','acddf','ggg']], "unknown_module")
	])
	def test_get_module_name(self, stats_list, expected):
		"""Test if module name is returned correctly"""
		result = profile_code.get_module_name(stats_list)
		self.assertEqual(result, expected)


	def test_run_from_command_line(self):
		"""Test if main function prints stats and saves to csv file correctly"""
		profile_code.run_from_command_line("./tests/fixture/fixture.prof")

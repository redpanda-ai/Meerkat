"""Unit test for meerkat.build_backend_layer"""

import unittest
import sys
import meerkat.build_backend_layer as builder

from meerkat.custom_exceptions import InvalidArguments
from nose_parameterized import parameterized

class Master:
	def __init__(self, private_ip_address):
		self.private_ip_address = private_ip_address

class BuildBackendLayerTests(unittest.TestCase):
	"""Our UnitTest class."""

	def test_run_ssh_commands__missing_pem_file(self):
		""" run_ssh_commands test that pem file is missing """
		fix = {
			"key_file" : "tests/missing.pem",
			"command_list" : ["uname"]
		}
		self.assertRaises(Exception, builder.run_ssh_commands, 
			"localhost", fix, "command_list", login="ubuntu")

	def test_run_ssh_commands__two_commands(self):
		""" run_ssh_commands test that two commands are run """
		fix = {
			"key_file" : "tests/test.pem",
			"command_list" : ["uname", "whoami"]
		}
		expected = ["['Linux\\n']", "['ubuntu\\n']"]
		result = builder.run_ssh_commands("localhost", fix, "command_list", login="ubuntu")
		self.assertEqual(expected, result)

	def test_run_ssh_commands__normal_use(self):
		""" run_ssh_commands test that a command is run """
		fix = {
			"key_file" : "tests/test.pem",
			"command_list" : ["uname"]
		}
		expected = ["['Linux\\n']"]
		result = builder.run_ssh_commands("localhost", fix, "command_list", login="ubuntu")
		self.assertEqual(expected, result)

	def test_run_ssh_commands__no_commands(self):
		""" run_ssh_commands test that command list is empty """
		fix = {
			"key_file" : "tests/test.pem",
		}
		self.assertRaises(KeyError, builder.run_ssh_commands, 
			"localhost", fix, "command_list", login="ubuntu")

	def test_initialize__normal_case(self):
		""" Assert that params is initialized """
		sys.argv = ["meerkat.build_backend_layer", "config/backend/1_c3_xlarge_az_a.json", "cluster_name"]
		result = builder.initialize()
		self.assertNotEqual(result, None)
		
	def test_initialize__invalid_arguments_number(self):
		""" Assert that when arguments number is invalid, InvalidArguments Exception is thrown """
		sys.argv = ["meerkat.build_backend_layer", "cluster_name"]
		self.assertRaises(InvalidArguments, builder.initialize)

	@parameterized.expand([
		(1, 0, 2, ["instance0", "instance1", "instance2"], 
			(["instance0"], [], ["instance1", "instance2"])),
		(0, 1, 0, ["instance0"], ([], ["instance0"], []))
	])
	def test_get_instance_lists_parameterized(self, masters, hybrids, slaves, instance_list, expected):
		"""Assert that instance lists are built correctly."""
		fix = {
			"instance_layout" : {
				"masters" : masters,
				"hybrids" : hybrids,
				"slaves" : slaves
			},
			"instances" : instance_list
		}
		builder.get_instance_lists(fix)
		self.assertEqual(fix["masters"], expected[0])
		self.assertEqual(fix["hybrids"], expected[1])
		self.assertEqual(fix["slaves"], expected[2])


	def test_get_instance_lists__empty_layout(self):
		""" Assert that an empty list of instances are generated """
		fix = {
			"instance_layout" : {
				"masters" : 0,
				"hybrids" : 0,
				"slaves" : 0
			},
			"instances" : []
		}
		builder.get_instance_lists(fix)
		self.assertEqual(fix["masters"], [])
		self.assertEqual(fix["hybrids"], [])
		self.assertEqual(fix["slaves"], [])

	def test_get_instance_lists__invalid_count_numbers(self):
		""" Assert that when layout sum isn't equal to instances number, ValueError is thrown """
		fix = {
			"instance_layout" : {
				"masters" : 1,
				"hybrids" : 1,
				"slaves" : 1
			},
			"instances" : ["instance0"]
		}
		self.assertRaises(ValueError, builder.get_instance_lists, fix)

	def test_get_master_ip_list__normal_case(self):
		""" Assert that a list of masters are created """
		master0 = Master("localhost")
		master1 = Master("52.34.28.58")
		fix = {
			"masters" : [master0, master1]
		}		
		expected = '"localhost", "52.34.28.58"'
		builder.get_master_ip_list(fix)
		self.assertEqual(expected, fix["master_ip_list"])

	def test_get_master_ip_list__empty_masters_list(self):
		""" Assert that an empty list of masters are created """
		fix = {
			"masters" : []
		}		
		expected = ''
		builder.get_master_ip_list(fix)
		self.assertEqual(expected, fix["master_ip_list"])

	@parameterized.expand([
		([Master("1.2.3.4"), Master("5.6.7.8")], 1, '"1.2.3.4"'),
		([], 1, ''),
		([Master("1.2.3.4"), Master("5.6.7.8"), Master("localhost")], 2, '"1.2.3.4", "5.6.7.8"')
	])
	def test_get_unicast_hosts__parameterized(self, instances, min_nodes, expected):
		""" Assert that a list of unicast hosts are created """
		fix = {
			"instances" : instances,
			"minimum_master_nodes" : min_nodes
		}
		builder.get_unicast_hosts(fix)
		self.assertEqual(fix["unicast_hosts"], expected)

if __name__ == '__main__':
	unittest.main()

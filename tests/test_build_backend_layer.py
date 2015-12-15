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

	@parameterized.expand([
		("tests/test.pem", ["uname", "whoami"], ["['Linux\\n']", "['ubuntu\\n']"]),
		("tests/test.pem", ["uname"], [ "['Linux\\n']" ]),
		("tests/test.pem", None, KeyError),
		("tests/missing.pem", None, Exception),
	])
	def test_run_ssh_commands__parameterized(self, key_file, command_list, expected):
		"""Tests the run_ssh_commands function with parameters."""
		host, cl, login = "localhost", "command_list", "ubuntu"
		fix = {
			"key_file" : key_file
		}
		if command_list is not None :
			fix["command_list"] = command_list
		if isinstance(expected, list):
			result = builder.run_ssh_commands(host, fix, cl, login=login)
			self.assertEqual(expected, result)
		else:
			self.assertRaises(expected, builder.run_ssh_commands, host, fix, cl, login=login)
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
		(1, 0, 2, ["instance0", "instance1", "instance2"], (["instance0"],[],["instance1", "instance2"])),
		(0, 1, 0, ["instance0"], ([],["instance0"],[])),
		(0, 0, 0, [], ([], [], [])),
		(1, 1, 1, ["instance0"], ValueError),
		(1, 1, 1, ["instance0", "instance1"], ValueError)
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
		if isinstance(expected, tuple):
			builder.get_instance_lists(fix)
			self.assertEqual(fix["masters"], expected[0])
			self.assertEqual(fix["hybrids"], expected[1])
			self.assertEqual(fix["slaves"], expected[2])
		else:
			self.assertRaises(expected, builder.get_instance_lists, fix)

	@parameterized.expand([
		(["localhost", "52.34.28.58"], '"localhost", "52.34.28.58"'),
		([], '')
	])
	def test_get_master_ip_list__parameterized(self, masters, expected):
		""" Assert that a list of masters are created """
		master_list = []
		for m in masters:
			master_list.append(Master(m))
		fix = { "masters" : master_list }
		builder.get_master_ip_list(fix)
		self.assertEqual(expected, fix["master_ip_list"])

#	@parameterized.expand([
#		(["meerkat_private", "s3cmd", "github"], ["sg_1"], ["meerkat_private", "s3cmd", "github", "sg_1"])
#	])
#	def test_confirm_security_groups__parameterized(self, security_groups, cluster_name, expected):
#		"""Assert that security groups are correctly processed."""
#		fix = { "security_group" : security_groups }
#		builder.confirm_security_groups(cls.conn, fix)
#		self.assertEqual(params["all_security_groups"], expected)

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

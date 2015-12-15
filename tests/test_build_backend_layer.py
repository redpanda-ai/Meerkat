"""Unit test for meerkat.build_backend_layer"""

import boto
import unittest
import uuid
import requests
from boto.ec2.connection import EC2Connection
import sys
import meerkat.build_backend_layer as builder
from meerkat.custom_exceptions import InvalidArguments
from nose_parameterized import parameterized

class Master:
	def __init__(self, private_ip_address):
		self.private_ip_address = private_ip_address

def delete_security_group(ec2_conn, sg_id, vpc_id):
	'delete A SINGLE security group that serves for unit tests'
	my_filter = {'group-name': sg_id, 'vpc-id': vpc_id}
	security_group = ec2_conn.get_all_security_groups(filters=my_filter)
	if len(security_group) != 0:
		security_group[0].delete()

def delete_instance(ec2_conn, instance_id):
	ec2_conn.terminate_instances(instance_ids = [instance_id])

def get_current_instance_public_ip():
	r = requests.get(r'http://jsonip.com')
	return r.json()['ip']

class BuildBackendLayerTests(unittest.TestCase):
	"""Our UnitTest class."""
	@classmethod
	def setUpClass(cls):
		'create necessary objects that might be used in multiple test cases'
		cls.default_region, cls.default_vpc_id = 'us-west-2', 'vpc-a930dacc'
		cls.my_region = boto.ec2.get_region(cls.default_region)
		cls.ec2_conn = EC2Connection(region=cls.my_region)
		cls.hvm_other_linux_ami = 'ami-f5e4a3c5'
		cls.default_sg = cls.ec2_conn.get_all_security_groups(filters=
			{'group-name': 'default', 'vpc-id': cls.default_vpc_id})[0]
		cls.default_availability_zone = 'us-west-2b'
		cls.default_subnet_id = 'subnet-e58d7580'
		cls.default_ip = get_current_instance_public_ip()
		cls.default_instance = cls.ec2_conn.get_all_instances(filters=
			{'ip_address':cls.default_ip})[0].instances[0]

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


	"""
	def test_acquire_instances__normal_use(self):
		params = {'all_security_groups':self.default_sg, 'key_name': 'jkey',
			'instance_layout': {'hybrids':1, 'masters':0, 'slaves':0},
			'ami-id': self.hvm_other_linux_ami, 'instance_type': 't2.micro',
			'placement': self.default_availability_zone, 'subnet-id':
			self.default_subnet_id}
		expected_owner_id = '003144629351'
		result = builder.acquire_instances(self.ec2_conn, params)
		self.assertEqual(expected_owner_id, result.owner_id)
		self.assertEqual(1,len(result.instances))
		self.assertEqual(self.default_sg, result.groups[0])
		delete_instance(self.ec2_conn, result.instance_ids)
	"""
	def test_confirm_security_groups__new_sg(self):
		test_sg_id = 'dummy_sg_' + str(uuid.uuid4())
		params = {'security_groups': [],
			'name': test_sg_id, "vpc-id": self.default_vpc_id}
		expected = 1
		builder.confirm_security_groups(self.ec2_conn, params)
		result = params['all_security_groups']
		self.assertEqual(expected,len(result))
		delete_security_group(self.ec2_conn, test_sg_id, self.default_vpc_id)

	def test_confirm_security_groups__no_new_sg(self):
		params = {'security_groups':[], 'name':'AWS-Elastica-Testing', 'vpc-id':
			self.default_vpc_id}
		expected = 1
		builder.confirm_security_groups(self.ec2_conn, params)
		result = params['all_security_groups']
		self.assertEqual(expected, len(result))

	def test_acquire_instances__wrong_virtualization(self):
		"""use a non-hvm ami"""
		params = {'all_security_groups':[self.default_sg], 'key_name': 'jkey',
			'instance_layout': {'hybrids':1, 'masters':0, 'slaves':0},
			'ami-id': 'ami-cf707eff', 'instance_type': 't2.micro',
			'placement': self.default_availability_zone, 'subnet-id':
			self.default_subnet_id}
		self.assertRaises(SystemExit, builder.acquire_instances, self.ec2_conn, params)

	def test_acquire_instances__wrong_subnet_id(self):
		"""use a subnet id default to us-west-2c"""
		params = {'all_security_groups':[self.default_sg], 'key_name': 'jkey',
			'instance_layout': {'hybrids':1, 'masters':0, 'slaves':0},
			'ami-id': self.hvm_other_linux_ami, 'instance_type': 't2.micro',
			'placement': self.default_availability_zone, 'subnet-id':
			'subnet-a092abe6'}
		self.assertRaises(SystemExit, builder.acquire_instances, self.ec2_conn, params)

	"""
	def test_copy_configuration_to_hosts__normal_use(self):
		params = {"key_file" : "tests/test.pem",
			'instances': [self.default_instance]}
		dst_file = 'tests/test.txt'
		builder.copy_configuration_to_hosts(params, dst_file, login='ubuntu')
	"""

	def test_create_security_group__normal_case(self):
		test_sg_id = 'dummy_sg_' + str(uuid.uuid4())
		expected = test_sg_id
		result = builder.create_security_group(self.ec2_conn, test_sg_id,
			 self.default_vpc_id)
		self.assertEqual(expected, result.name)
		delete_security_group(self.ec2_conn, test_sg_id, self.default_vpc_id)

	def test_create_security_group__using_existing_sg_name(self):
		test_sg_id = 'dummy_sg_' + str(uuid.uuid4())
		_ = builder.create_security_group(self.ec2_conn, test_sg_id, self.default_vpc_id)
		self.assertRaises(SystemExit, builder.create_security_group,
			self.ec2_conn, test_sg_id, self.default_vpc_id)
		delete_security_group(self.ec2_conn, test_sg_id, self.default_vpc_id)

	def test_create_security_group__invalid_vpc_id(self):
		test_sg_id = 'dummy_sg_' + str(uuid.uuid4())
		self.assertRaises(SystemExit, builder.create_security_group, self.ec2_conn,
			test_sg_id, 'invalid_vpc_id')

if __name__ == '__main__':
	unittest.main()

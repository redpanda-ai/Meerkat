"""Unit test for meerkat.various_tools"""
import boto
import meerkat.various_tools
import numpy as np
import unittest
import uuid
from boto.ec2.connection import EC2Connection
import meerkat.build_backend_layer as builder

def delete_security_group(ec2_conn, sg_id, vpc_id):
	'delete A SINGLE security group that serves for unit tests'
	my_filter = {'group-name': sg_id, 'vpc-id': vpc_id}
	security_group = ec2_conn.get_all_security_groups(filters=my_filter)
	if len(security_group) != 0:
		security_group[0].delete()

def delete_instance(ec2_conn, instance_id):
	'delelet A SINGLE instance that serves for unit tests'
	ec2_conn.terminate_instances(instance_ids=[instance_id])

class BuildBackendLayerTests(unittest.TestCase):
	"""Our UnitTest class."""

	@classmethod
	def setUpClass(cls):
		"""create necessary objects that might be used in multiple test cases"""
		cls.default_region, cls.default_vpc_id = 'us-west-2', 'vpc-a930dacc'
		cls.my_region = boto.ec2.get_region(cls.default_region)
		cls.ec2_conn = EC2Connection(region=cls.my_region)
		cls.hvm_other_linux_ami = 'ami-f5e4a3c5'
		cls.default_subnet_id = 'subnet-e58d7580'
		cls.default_availability_zone = 'us-west-2b'
		cls.default_sg = cls.ec2_conn.get_all_security_groups(
			filters={'group-name':'default', 'vpc-id': cls.default_vpc_id})

	"""
	def test_acquire_instances__normal_use(self):
		
		This test creates a new instance and security group every time 
		we run it. It finishes when the new instance is running. This test
		cost about $0.01 - $0.26, depending on the instance type
		
		params = {'ami-id': self.hvm_other_linux_ami, 'key_name': 'jkey',
			'instance_type': 't2.micro', 'placement': self.default_availability_zone,
			'subnet-id': self.default_subnet_id, 'instance_layout': {'hybrids': 1,
			'masters': 0, 'slaves': 0}, 'all_security_groups': self.default_sg}
		expected_owner_id = '003144629351'
		reservation = builder.acquire_instances(self.ec2_conn, params)
		self.assertEqual(reservation.owner_id, expected_owner_id)
		delete_instance(self.ec2_conn, reservations.instances[0].id)
	"""
	def test_acquire_instances__wrong_virtualization(self):
		"""using a non-hvm ami to create a t2.micro instance"""
		params = {'ami-id': 'ami-cf707eff', 'key_name': 'jkey',
			'instance_type': 't2.micro', 'placement': self.default_availability_zone,
			'subnet-id': self.default_subnet_id, 'instance_layout': {'hybrids': 1,
			'masters': 0, 'slaves': 0}, 'all_security_groups': self.default_sg}
		self.assertRaises(SystemExit, builder.acquire_instances, self.ec2_conn, params)


	def test_create_security_group__normal_use(self):
		test_sg_id = 'dummy_sg_' + str(uuid.uuid4())
		expected = test_sg_id
		result = builder.create_security_group(self.ec2_conn, test_sg_id,
			 self.default_vpc_id)
		self.assertEqual(expected, result.name)
		delete_security_group(self.ec2_conn, test_sg_id, self.default_vpc_id)

	def test_create_security_group__using_existing_sg_name(self):
		test_sg_id = 'dummy_sg_' + str(uuid.uuid4())
		_ = builder.create_security_group(self.ec2_conn, test_sg_id,
			 self.default_vpc_id)
		self.assertRaises(SystemExit, builder.create_security_group,
			self.ec2_conn, test_sg_id, self.default_vpc_id)
		delete_security_group(self.ec2_conn, test_sg_id, self.default_vpc_id)

	def test_create_security_group__invalid_vpc_id(self):
		test_sg_id = 'dummy_sg_' + str(uuid.uuid4())
		self.assertRaises(SystemExit, builder.create_security_group, self.ec2_conn,
			test_sg_id, 'invalid_vpc_id')

	def test_run_ssh_commands__missing_pem_file(self):
		fix = {
			"key_file" : "tests/missing.pem",
			"command_list" : ["uname"]
		}
		self.assertRaises(Exception, builder.run_ssh_commands, "localhost", fix, "command_list", login="ubuntu")

	def test_run_ssh_commands__two_commands(self):
		fix = {
			"key_file" : "tests/test.pem",
			"command_list" : ["uname", "whoami"]
		}
		expected = ["['Linux\\n']", "['ubuntu\\n']"]
		result = builder.run_ssh_commands("localhost", fix, "command_list", login="ubuntu")
		self.assertEqual(expected, result)

	def test_run_ssh_commands__normal_use(self):
		fix = {
			"key_file" : "tests/test.pem",
			"command_list" : ["uname"]
		}
		expected = [ "['Linux\\n']" ]
		result = builder.run_ssh_commands("localhost", fix, "command_list", login="ubuntu")
		self.assertEqual(expected, result)

	def test_run_ssh_commands__no_commands(self):
		fix = {
			"key_file" : "tests/test.pem",
		}
		self.assertRaises(KeyError, builder.run_ssh_commands, "localhost", fix, "command_list", login="ubuntu")

if __name__ == '__main__':
	unittest.main()

import boto
import json
import paramiko
import sys
import time

from boto.ec2.connection import EC2Connection
from boto.regioninfo import RegionInfo
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping

#AWS_SECRET_ACCESS_KEY="0432+/xrV6cpnc2N7F5VpJbhKhXBuxIJpYwM3bQl"
#AWS_ACCESS_KEY_ID="AKIAIHJQDVN46QEUHU6A"

def connect_to_ec2(region):
	"""Returns a connection to EC2"""
	try:
		conn = boto.ec2.connect_to_region(region, aws_access_key_id=AWS_ACCESS_KEY_ID,
			aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
	except:
		print("Error connecting to EC2, check your credentials")
		sys.exit()
	return conn

def confirm_security_groups(conn, params):
	existing_groups = params["security_groups"]
	existing_group_count = len(existing_groups)
	security_groups = conn.get_all_security_groups()
	groups_found = 0
	new_group_found = False
	all_groups = []
	for group in security_groups:
		if group.name in existing_groups:
			print("Security group {0} found, continuing".format(group))
			groups_found +=1
			all_groups.append(group)
		elif group.name == params["name"]:
			new_group_found = True
			all_groups.append(group)
	if groups_found == existing_group_count:
		print("All pre-existing groups found, continuing".format(group))
	if not new_group_found:
		print("Adding group {0}".format(params["name"]))
		all_groups.append(create_security_group(conn, params["name"]))
	for group in all_groups:
		print(group)
	params["all_security_groups"] = all_groups

def create_security_group(conn, name):
	try:
		my_security_group = conn.create_security_group(name, name)
		print("Added {0}".format(my_security_group))
		return my_security_group
	except boto.exception.EC2ResponseError:
		print("Unable to create {0}, aborting".format(name))
		sys.exit()

def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None
	if len(sys.argv) != 2:
		usage()
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)
	try:
		input_file = open(sys.argv[1], encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.error("%s not found, aborting.", sys.argv[1])
		sys.exit()
	return params

def acquire_instances(ec2_conn, params):
	print("Acquiring instances")
	#set up block device mapping
	bdm = BlockDeviceMapping()
	eph0 = BlockDeviceType()
	eph1 = BlockDeviceType()
	eph0.ephemeral_name = 'ephemeral0'
	eph1.ephemeral_name = 'ephemeral1'
	bdm['/dev/sdb'] = eph0
	bdm['/dev/sdc'] = eph1
	reservations = ec2_conn.run_instances(params["ami-id"],
		key_name=params["key_name"], instance_type=params["instance_type"],
		placement=params["placement"], block_device_map=bdm,
		min_count=params["instance_count"], max_count=params["instance_count"],
		security_groups=params["all_security_groups"])

	print("Reservations {0}".format(reservations))

	print("Waiting for instances to start...")
	count_runners = 0
	while count_runners < params["instance_count"]:
		instances = reservations.instances
		count_runners = 0
		for i in instances:
			private_ip_address = i.private_ip_address
			id = i.id
			state = i.update()
			print("IP: {0}, ID: {1}, State: {2}".format(private_ip_address, id, state))
			if state == "running":
				count_runners += 1
		if count_runners < params["instance_count"]:
			print("Still waiting on {0} instances...".format(params["instance_count"] - count_runners))
			time.sleep(10)
	print("All instances running.")

#	instance = reservations.instances[0]
#	status = instance.update()
#	while status == 'pending':
#		time.sleep(10)
#		status = instance.update()
#	if status == 'running':
#		print("New instance {0} accessible.".format(instance.id))
#	else:
#		print("Instance status: {0}".format(status))

def map_block_devices(ec2_conn, params):
	bdm = BlockDeviceMapping()
	eph0 = BlockDeviceType(ephemeral_name = 'ephemeral0')
	eph1 = BlockDeviceType(ephemeral_name = 'ephemeral1')
	bdm['/dev/sdb'] = eph0
	bdm['/dev/sdc'] = eph1
	print("Map is {0}".format(bdm))
	params["bdm"] = bdm

def start():
	params = initialize()
	my_region = boto.ec2.get_region(params["region"])
	ec2_conn = EC2Connection(region=my_region)
	confirm_security_groups(ec2_conn, params)
	map_block_devices(ec2_conn, params)
	#print(params["mapping"])
	acquire_instances(ec2_conn, params)

start()

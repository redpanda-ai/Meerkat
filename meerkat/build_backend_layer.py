#!/usr/local/bin/python3.3

"""This module creates an Elasticsearch cluster on Amazon EC2.

Created on Sep 30, 2014
@author: J. Andrew Key
"""

#################### USAGE ##########################

# python3.3 -m meerkat.build_backend_layer [config_file] [cluster_name]
# python3.3 -m meerkat.build_backend_layer config/stack.json Meerkat-ES

#####################################################

import boto
import fileinput
import json
import paramiko
import shutil
import sys
import time
import logging

from elasticsearch import Elasticsearch
from boto.ec2.connection import EC2Connection
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.exception import EC2ResponseError
from .custom_exceptions import InvalidArguments

def acquire_instances(ec2_conn, params):
	"""Requests EC2 instances from Amazon and monitors the request until
	it is fulfilled.."""
	print("Acquiring instances")
	sec_groups = params["all_security_groups"]
	sec_group_ids = []
	for group in sec_groups:
		sec_group_ids.append(group.id)
	print("Security group ids {0}".format(sec_group_ids))
	#set up block device mapping
	bdm = BlockDeviceMapping()
	eph0, eph1 = BlockDeviceType(), BlockDeviceType()
	eph0.ephemeral_name, eph1.ephemeral_name = 'ephemeral0', 'ephemeral1'
	bdm['/dev/sdb'], bdm['/dev/sdc'] = eph0, eph1
	layout = params["instance_layout"]
	instance_count = layout["masters"] + layout["hybrids"] + layout["slaves"]
	reservations = None
	try:
		reservations = ec2_conn.run_instances(params["ami-id"],\
			key_name=params["key_name"], instance_type=params["instance_type"],\
			placement=params["placement"], block_device_map=bdm,\
			min_count=instance_count, max_count=instance_count,\
			subnet_id=params["subnet-id"], security_group_ids=sec_group_ids)
	except EC2ResponseError as err:
		print("Error getting instances, aborting: Exception {0}".format(err))
		print("Unexpected error:", sys.exc_info()[0])
		print(reservations)
		sys.exit()
	poll_pending_instances(reservations, instance_count, params)

def assign_ebs_volumes(ec2_conn, params):
	"""Assigns ebs volumes to instances."""
	get_snapshot(ec2_conn, params)
	snapshot = params["snapshot"]
	print("Snapshot: {0}".format(params["snapshot"]))
	placement = params["placement"]
	instances = params["instances"]
	for i in instances:
		max_attempts = 60
		#Try multiple times to attach a volume
		for j in range(0, max_attempts):
			try:
				if j > 0:
					print("Making attempt {0} of {1} to attach volume.".format(\
						j, max_attempts))
					print("Placement {0}, Snapshot {1}".format(placement, snapshot))
					new_volume = ec2_conn.create_volume(100, placement,\
						snapshot=snapshot, volume_type="gp2")
					print("New volume created {0}, {1}".format(new_volume.id,\
						new_volume.status))
					attach_volume_to_instance(ec2_conn, new_volume, i, params)
					break
			except Exception as err:
				j += 1
				print("Exception {0}".format(err))
				print("Unexpected error:", sys.exc_info()[0])
				print("Attempt #{0} in 3 seconds.".format(j))
				time.sleep(3)
	print("All volumes created from snapshots and mounted.")

def attach_volume_to_instance(ec2_conn, volume, instance, params):
	"""Attaches EBS volumes to EC2 instances."""
	print("In attach_volume_to_instance")
	status, tries, max_tries = volume.status, 0, 60
	while (status != "available") and (tries < max_tries):
		print("Volume status {0}".format(status))
		time.sleep(5)
		tries += 1
		print("Updating")
		volume.update()
		status = volume.volume_state()
	print("Volume available")
	my_ebs_mount = params["ebs_mount"]
	max_attempts = 60
	print("Volume.id {0}, Instance.id {1}, my_ebs_mount {2}".format(\
		volume.id, instance.id, my_ebs_mount))
	status_line = "volume.id {0}, instance.id {1}, device_name {2}, "\
		"attachment_state {3}"
	for j in range(0, max_attempts):
		try:
			print("Trying to attach volume")
			_ = ec2_conn.attach_volume(volume.id, instance.id, my_ebs_mount)
			print("Trying to update volume stats")
			volume.update()
			print("Trying to read volume state")
			status = volume.attachment_state()
			print("Volume attachment state {0}".format(status))
			if status == 'attaching':
				print(status_line.format(volume.id, instance.id, my_ebs_mount, status))
				j = max_attempts
				return
			time.sleep(3)
			continue
		except boto.exception.EC2ResponseError:
			print(status_line.format(volume.id, instance.id, my_ebs_mount, "attached"))
			j = max_attempts
			return
		except Exception as err:
			j += 1
			print("Exception {0}".format(err))
			print("Unexpected error:", sys.exc_info()[0])
			print("Attempt #{0} in 3 seconds.".format(j))
			time.sleep(3)

def configure_servers(params):
	"""Creates all configuration files and uploads the configuration needed on each
	instance."""
	# __UNICAST HOSTS -> "172.31.1.191", "172.31.1.192"
	# __MINIMUM_MASTER_NODES -> params["minimum_master_nodes"]
	# __NODE_NAME -> params["name"] + a number
	print("Creating config files from templates.")
	get_unicast_hosts(params)
	get_master_ip_list(params)
	yml_file = customize_config_file(params, "elasticsearch_yml")

	print("Copying configuration template to hosts.")
	copy_configuration_to_hosts(params, yml_file)


def confirm_security_groups(conn, params):
	"""Confirms that the security groups we need for accessng our cluster
	are correctly in place"""
	existing_groups = params["security_groups"]
	existing_group_count = len(existing_groups)
	security_groups = conn.get_all_security_groups()
	groups_found = 0
	new_group_found = False
	all_groups = []
	for group in security_groups:
		if group.name in existing_groups:
			print("Security group {0} found, continuing".format(group))
			groups_found += 1
			all_groups.append(group)
		elif group.name == params["name"]:
			new_group_found = True
			all_groups.append(group)
	if groups_found == existing_group_count:
		print("All pre-existing groups found, continuing".format(group))
	if not new_group_found:
		print("Adding group {0}".format(params["name"]))
		#Need to add vpc_id as named parameters
		all_groups.append(create_security_group(conn, params["name"],\
			params["vpc-id"]))
	for group in all_groups:
		print(group)
	params["all_security_groups"] = all_groups

def connect_to_ec2(region):
	"""Returns a connection to EC2"""
	try:
		conn = boto.ec2.connect_to_region(region,\
			aws_access_key_id=AWS_ACCESS_KEY_ID,\
			aws_secret_access_key=AWS_SECRET_ACCESS_KEY, debug=2)
	except:
		print("Error connecting to EC2, check your credentials")
		sys.exit()
	return conn

def copy_configuration_to_hosts(params, dst_file):
	"""Copies configuration files to each instance in your ES cluster."""
	rsa_private_key_file = params["key_file"]
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	for instance in params["instances"]:
		instance_ip_address = instance.private_ip_address
		print("Pushing config file to {0}".format(instance_ip_address))
		ssh.connect(instance_ip_address, username="root",\
			key_filename=rsa_private_key_file)
		sftp = ssh.open_sftp()
		#ALERT hard-coded string
		sftp.put(dst_file, "/etc/elasticsearch/elasticsearch.yml")
		ssh.close()

def create_security_group(conn, name, vpc):
	"""Creates a new security group for our cluster."""
	try:
		my_security_group = conn.create_security_group(name, name, vpc)
		print("Added {0}".format(my_security_group))
		return my_security_group
	except boto.exception.EC2ResponseError:
		print("Unable to create {0}, aborting".format(name))
		sys.exit()

def customize_config_file(params, src_file):
	"""Duplicates a configuration file template and then customizes it."""
	templates = params["template_files"]
	src_file = templates["path"] + templates[src_file]
	dst_file = src_file + "." + params["name"]
	try:
		shutil.copy(src_file, dst_file)
	except shutil.Error as err:
		print("Error copying template. {0}".format(err))
	except IOError:
		print("IO Error copying template.")

	with fileinput.input(files=(dst_file), inplace=True) as altered_file:
		for line in altered_file:
			line = line.strip().replace("__UNICAST_HOSTS",\
				params["unicast_hosts"])
			line = line.replace("__MINIMUM_MASTER_NODES",\
				str(params["minimum_master_nodes"]))
			line = line.strip().replace("__MASTER_IP_LIST",\
				params["master_ip_list"])
			print(line)

	return dst_file

def get_master_ip_list(params):
	"""Creates a comma separated list of masters for use in configuration files."""
	masters = params["masters"]
	master_private_ips = []
	for master in masters:
		master_private_ips.append('"' + master.private_ip_address + '"')
	params["master_ip_list"] = ', '.join(master_private_ips)
	print("master_ip_list {0}".format(params["master_ip_list"]))

def get_instance_lists(params):
	"""Generates a list of instances that will be stored in the params dict."""
	layout = params["instance_layout"]
	master_count = layout["masters"]
	hybrid_count = layout["hybrids"]
	slave_count = layout["slaves"]
	print("Masters: {0}, Hybrids: {1}, Slaves: {2}".format(master_count,\
		hybrid_count, slave_count))
	#Split the instances into separate lists
	step = 0
	params["masters"] = params["instances"][step:step + master_count]
	step += master_count
	params["hybrids"] = params["instances"][step:step + hybrid_count]
	step += hybrid_count
	params["slaves"] = params["instances"][step:step + slave_count]

	print("Masters {0}".format(params["masters"]))
	print("Hybrids {0}".format(params["hybrids"]))
	print("Slaves {0}".format(params["slaves"]))

def get_snapshot(ec2_conn, params):
	"""Gets the EBS snapshot from Amazon"""
	print("Assigning EBS volumes")
	ebs_mapping = params["ebs_mapping"]
	max_attempts, sleep_between_attempts = 6, 10
	snapshots = None
	for j in range(0, max_attempts):
		for item in ebs_mapping:
			try:
				snapshot_id = item["snapshot-id"]
				params["ebs_mount"] = item["mount"]
				snapshot_ids = [snapshot_id]
				snapshots = ec2_conn.get_all_snapshots(snapshot_ids=snapshot_ids)
				snapshot = snapshots[0]
			except Exception as err:
				j += 1
				print("Unexpected error:", sys.exc_info()[0])
				print("Exception {0}".format(err))
				print("Attempt #{0} in {1} seconds.".format(j, sleep_between_attempts))
				time.sleep(sleep_between_attempts)
		break
	params["snapshot"] = snapshot

def get_unicast_hosts(params):
	"""Creates a list of unicast hosts, which will be 'master' nodes"""
	result = ''
	hosts = params["instances"][0:params["minimum_master_nodes"]]
	for host in hosts:
		result += '"' + host.private_ip_address + '", '
	params["unicast_hosts"] = result[:-2]

def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None
	if len(sys.argv) != 3:
		print("Supply the following arguments: config_file, cluster-name")
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)
	try:
		input_file = open(sys.argv[1], encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
		params["name"] = sys.argv[2]
	except IOError:
		logging.error("%s not found, aborting.", sys.argv[1])
		sys.exit()
	return params

def map_block_devices(params):
	"""Creates block device mappings and adds them to our params dict."""
	bdm = BlockDeviceMapping()
	eph0 = BlockDeviceType(ephemeral_name='ephemeral0')
	eph1 = BlockDeviceType(ephemeral_name='ephemeral1')
	bdm['/dev/sdb'] = eph0
	bdm['/dev/sdc'] = eph1
	print("Map is {0}".format(bdm))
	params["bdm"] = bdm

def poll_for_cluster_status(params):
	"""Poll for cluster status until all nodes are 'green'"""
	cluster_nodes = [params["instances"][0].private_ip_address]
	print("Polling for cluster status green")
	max_attempts, sleep_between_attempts = 30, 10
	#Try multiple times to get cluster health of green
	target_nodes = len(params["instances"])
	status_line = "Status: {0}, Number of Nodes: {1}, Target Nodes: {2}"
	for j in range(0, max_attempts):
		try:
			if j > 0:
				print("Making attempt {0} of {1} for cluster status.".format(j,\
					max_attempts))
				try:
					es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,\
						sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)
				except Exception as err:
					print("Exception while trying to make Elasticsearch connection.")
					raise("Error trying to connect to ES node.")
				print("Cluster is online.")
				print("Attempting to poll for health.")
				status, number_of_nodes = "unknown", 0
				while (status != "green") or (number_of_nodes < target_nodes):
					result = es_connection.cluster.health()
					if result:
						status = result["status"]
						number_of_nodes = result["number_of_nodes"]

						print(status_line.format(status, number_of_nodes,\
							target_nodes))
						time.sleep(sleep_between_attempts)
					else:
						time.sleep(sleep_between_attempts)
				break
			if j >= max_attempts:
				print("Error getting cluster status, aborting abnormally.")
				sys.exit()
		except Exception as err:
			j += 1
			print("Exception {0}".format(err))
			print("Attempt #{0} in {1} seconds.".format(j, sleep_between_attempts))
			time.sleep(sleep_between_attempts)
	print("Congratulations your cluster is fully operational.")

def poll_pending_instances(reservations, instance_count, params):
	"""Poll pending instances until all are running."""
	print("Reservations {0}".format(reservations))
	print("Waiting for instances to start...")
	count_runners = 0
	while count_runners < instance_count:
		instances = reservations.instances
		count_runners = 0
		max_attempts = 6
		for i in instances:
			private_ip_address = i.private_ip_address
			instance_id = i.id
			#Try multiple times to get instance update if necessary
			for j in range(0, max_attempts):
				if j >= max_attempts:
					print("Error updating instance state, aborting abnormally.")
					sys.exit()
				try:
					if j > 0:
						print("Attempt {0} of {1} for instance update.".format(\
							j, max_attempts))
					state = i.update()
				except:
					j += 1
					print("Attempt #{0} in 3 seconds.".format(j))
					time.sleep(3)
			#Display state for this instances
			print("IP: {0}, ID: {1}, State: {2}".format(private_ip_address,\
				instance_id, state))
			#Increment count if running
			if state == "running":
				count_runners += 1

		pending_instances = instance_count - count_runners
		if count_runners < instance_count:
			print("Waiting on {0} instances...".format(pending_instances))
			time.sleep(10)
	print("All instances running.")
	params["instances"] = instances

def run_ssh_commands(instance_ip_address, params, command_list):
	"""Issues a list of commands via SSH to an EC2 instance."""
	rsa_private_key_file = params["key_file"]
	shell_commands = params[command_list]
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	try:
		ssh.connect(instance_ip_address, username="root",\
			key_filename=rsa_private_key_file)
	except Exception as err:
		print("Exception trying to ssh into host {0}".format(err))
		raise Exception("Error trying to ssh into host.")

	for item in shell_commands:
		print(item)
		_, stdout, stderr = ssh.exec_command(item)
		my_stdout = str(stdout.readlines())
		my_stderr = str(stderr.readlines())
		if my_stdout != "[]":
			print(my_stdout)
		if my_stderr != "[]":
			print(my_stderr)
	ssh.close()

def send_shell_commands(params, command_set, instance_list):
	"""Sends a list of shell commands, to each instance, displaying the result."""
	#Try multiple times to get instance update if necessary
	max_attempts = 6
	for j in range(0, max_attempts):
		try:
			if j >= 0:
				print("Making attempt {0} of {1} for ssh access.".format(j, max_attempts))
			for instance in params[instance_list]:
				run_ssh_commands(instance.private_ip_address, params, command_set)
			break
		except:
			j += 1
			print("Attempt #{0} in 10 seconds.".format(j))
			time.sleep(10)

def start():
	"""This function runs the entire sequence needed to build a cluster."""
	logging.basicConfig(filename='boto.log', level=logging.DEBUG)
	params = initialize()
	my_region = boto.ec2.get_region(params["region"])
	ec2_conn = EC2Connection(region=my_region)
	print("Connection established.")
	confirm_security_groups(ec2_conn, params)
	map_block_devices(params)
	acquire_instances(ec2_conn, params)
	if "ebs_mapping" in params:
		assign_ebs_volumes(ec2_conn, params)
	get_instance_lists(params)
	send_shell_commands(params, "mount_data_commands", "instances")
	configure_servers(params)
	send_shell_commands(params, "configure_master_commands", "masters")
	send_shell_commands(params, "configure_hybrid_commands", "hybrids")
	send_shell_commands(params, "configure_slave_commands", "slaves")
	poll_for_cluster_status(params)

#Main program
start()

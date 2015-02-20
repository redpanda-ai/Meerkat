#!/usr/local/bin/python3.3

"""This module monitors an Elasticsearch cluster on Amazon EC2.

Created on 2014-02-19
@author: J. Andrew Key
"""

#################### USAGE ####################################################
# python3.3 -m meerkat.prototype <configuration_file> <cluster-name>
# python3.3 -m meerkat.prototype config/prototype/config_1.json meerkat-key-6
###############################################################################

import boto
import fileinput
import json
import paramiko
import re
import shutil
import sys
import time
import logging

from elasticsearch import Elasticsearch
from boto.ec2.connection import EC2Connection
from boto.ec2.instance import Instance
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.exception import EC2ResponseError
from .custom_exceptions import InvalidArguments

def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None
	if len(sys.argv) != 3:
		logging.debug("Supply the following arguments: config_file, cluster-name")
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

def confirm_security_groups(conn, params):
	"""Confirms that the security groups we need for accessng our cluster
	are correctly in place"""
	existing_groups = params["security_groups"]
	existing_group_count = len(existing_groups)
	security_groups = conn.get_all_security_groups()
	logging.debug(security_groups)
	groups_found = 0
	new_group_found = False
	all_groups = []
	for group in security_groups:
		if group.name in existing_groups:
			logging.debug("Security group {0} found, continuing".format(group))
			groups_found += 1
			all_groups.append(group)
		elif group.name == params["name"]:
			new_group_found = True
			all_groups.append(group)
	if groups_found == existing_group_count:
		logging.debug("All pre-existing groups found, continuing".format(group))
	if not new_group_found:
		logging.critical("Group {0} not found, aborting".format(params["name"]))
		#TODO: Throw group not found Exception
		sys.exit()
	for group in all_groups:
		logging.debug(group)
	params["all_security_groups"] = all_groups

def poll_for_cluster_statistics(params):
	"""Poll for cluster status until all nodes are 'green'"""
	cluster_nodes = [params["instances"][0].private_ip_address]
	logging.debug("Polling for cluster status green")
	max_attempts, sleep_between_attempts = 30, 10
	#Try multiple times to get statistics
	target_nodes = len(params["instances"])
	status_line = "Status: {0}, Number of Nodes: {1}, Target Nodes: {2}"
	pattern = re.compile("inet..(.*)......")
	for j in range(0, max_attempts):
		try:
			if j > 0:
				logging.debug("Making attempt {0} of {1} for cluster statistics.".format(j,\
					max_attempts))
				try:
					es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,\
						sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)
				except Exception as err:
					logging.warning("Exception while trying to make Elasticsearch connection.")
					raise("Error trying to connect to ES node.")
				logging.debug("Attempting to collect statistics.")
				p = lambda x: json.dumps(x, sort_keys=False, indent=4, separators=(',', ': '))
				result = p(es_connection.nodes.stats(metric="name,os,thread_pool"))
				nodes = json.loads(result)["nodes"]
				trim_result = { "nodes": [] }
				high_cpu, high_search_queue = 0, 0
				for k in nodes.keys():
					ip = pattern.match(nodes[k]["ip"][0]).group(1),
					cpu = nodes[k]["os"]["cpu"]["usage"]
					search_queue = nodes[k]["thread_pool"]["search"]["queue"]
					if cpu > high_cpu:
						high_cpu = cpu
					if search_queue > high_search_queue:
						high_search_queue = search_queue
					node = {
						"ip" : ip,
						"cpu" : cpu,
						"search_queue" : search_queue
					}
					trim_result["nodes"].append(node)
				summary = { "high_cpu": high_cpu, "high_search_queue": high_search_queue }
					
				logging.debug("Returning statistics")
				return summary, p(trim_result), result
				#return result
			if j >= max_attempts:
				logging.warning("Error getting cluster statistics, aborting abnormally.")
				sys.exit()
		except Exception as err:
			j += 1
			logging.warning("Exception {0}".format(err))
			logging.warning("Attempt #{0} in {1} seconds.".format(j, sleep_between_attempts))
			time.sleep(sleep_between_attempts)

def poll_for_cluster_status(params):
	"""Poll for cluster status until all nodes are 'green'"""
	cluster_nodes = [params["instances"][0].private_ip_address]
	logging.debug("Polling for cluster status green")
	max_attempts, sleep_between_attempts = 30, 10
	#Try multiple times to get cluster health of green
	target_nodes = len(params["instances"])
	status_line = "Status: {0}, Number of Nodes: {1}, Target Nodes: {2}"
	for j in range(0, max_attempts):
		try:
			if j > 0:
				logging.debug("Making attempt {0} of {1} for cluster status.".format(j,\
					max_attempts))
				try:
					es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,\
						sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)
				except Exception as err:
					logging.warning("Exception while trying to make Elasticsearch connection.")
					raise("Error trying to connect to ES node.")
				logging.debug("Cluster found.")
				logging.debug("Attempting to poll for health.")
				status, number_of_nodes = "unknown", 0
				while (status != "green") or (number_of_nodes < target_nodes):
					result = es_connection.cluster.health()
					if result:
						status = result["status"]
						number_of_nodes = result["number_of_nodes"]
						#TODO: Return with vital information
						return status, number_of_nodes, target_nodes

						#logging.warning(status_line.format(status, number_of_nodes,\
						#	target_nodes))
						if (status != "green") or (number_of_nodes < target_nodes):
							time.sleep(sleep_between_attempts)
					else:
						time.sleep(sleep_between_attempts)
				break
			if j >= max_attempts:
				logging.warning("Error getting cluster status, aborting abnormally.")
				sys.exit()
		except Exception as err:
			j += 1
			logging.warning("Exception {0}".format(err))
			logging.warning("Attempt #{0} in {1} seconds.".format(j, sleep_between_attempts))
			time.sleep(sleep_between_attempts)
	logging.warning("Congratulations your cluster is fully operational.")

def recommend(params, summary):
	"""This function recommends scaling up or down."""
	p = lambda x: json.dumps(x, sort_keys=False, indent=4, separators=(',', ': '))
	judgment = None
	scaling_rules = params["scaling_rules"]
	down_cpu, down_queue = scaling_rules["down"]["cpu"], scaling_rules["down"]["search_queue"]
	up_cpu, up_queue = scaling_rules["up"]["cpu"], scaling_rules["up"]["search_queue"]
	node_min, node_max = params["nodes"]["minimum"], params["nodes"]["maximum"]
	running, not_running = len(params["running"]), len(params["not_running"])
	cpu, queue = summary["high_cpu"], summary["high_search_queue"]

	#logging.warning("IN-USE: {0}, STANDING-BY: {1}".format(running, not_running))

	if (cpu <= down_cpu) and (queue <= down_queue):
		if running >= node_min:
			judgment = "scale down"
		else:
			judgment = "WARNING: scale down limit reached"
	elif (cpu >= up_cpu) and (queue >= up_queue):
		if not_running > 0:
			judgment = "scale up"
		else:
			judgment = "WARNING: scale up limit reached"
	else:
		judgment = "neutral"
	logging.warning("IN-USE: {0}, STANDING-BY: {1}, CPU: {2}, QUEUE: {3}, RECOMMENDATION {4}".format(\
		running, not_running, cpu, queue, judgment))
	return judgment

def judge(params, ec2_conn):
	reservations = ec2_conn.get_all_instances()
	instances = []
	filtered_instances = []
	for reservation in reservations:
		instances.extend(reservation.instances)
	for instance in instances:
		for group in instance.groups:
			if group.name == params["name"]:
				filtered_instances.append(instance)
				break
	instances = [ i for i in filtered_instances if i.id not in params["master_instances"] ]
	params["running"] = [i for i in instances if i.state == 'running' ]
	params["not_running"] = [i for i in instances if i.state != 'running' ]
	params["instances"] = params["running"]
	status, active_nodes, data_nodes = poll_for_cluster_status(params)
	#status_line = "Status: {0}, Active Nodes: {1}, Data Nodes: {2}"
	#logging.warning(status_line.format(status, active_nodes, data_nodes))
	summary, trim_result, result = poll_for_cluster_statistics(params)
	cpu, queue = summary["high_cpu"], summary["high_search_queue"]
	judgment = recommend(params, summary)
	#logging.warning("CPU: {0}, QUEUE: {1}, RECOMMENDATION {2}".format(cpu, queue, judgment))

def start():
	"""This function starts the monitor."""
	logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/prototype.log', \
		level=logging.WARNING)
	console = logging.StreamHandler()
	console.setLevel(logging.WARNING)
	logging.getLogger('').addHandler(console)

	logging.debug("Advanced Infrastructure Monitor.")
	params = initialize()
	my_region = boto.ec2.get_region(params["region"])
	ec2_conn = EC2Connection(region=my_region)
	logging.debug("EC2Connection established.")
	confirm_security_groups(ec2_conn, params)
	interval = params["scaling_rules"]["interval"]
	count = 0
	while count < 20:
		judge(params, ec2_conn)
		time.sleep(interval)
		count += 1

#	for instance in instances:
#		logging.debug("ID {0}, state {1}, private_ip {2}".format(\
#			instance.id, instance.state, instance.private_ip_address))

#MAIN PROGRAM
start()

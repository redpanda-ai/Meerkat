#!/usr/local/bin/python3.3

"""This module monitors an Elasticsearch cluster on Amazon EC2.

Created on 2014-02-19
@author: J. Andrew Key
"""

#################### USAGE ####################################################
# python3.3 -m meerkat.auto_scaler <configuration_file> <cluster-name>
# python3.3 -m meerkat.auto_scaler config/auto_scaler/config_1.json my_cluster
###############################################################################

import boto
import fileinput
import json
import logging
import re
import sys
import time

from elasticsearch import Elasticsearch
from boto.ec2.connection import EC2Connection
from boto.exception import EC2ResponseError

from .custom_exceptions import InvalidArguments

def confirm_security_groups(conn, params):
	"""Confirms that the security groups we need for accessng our cluster
	are correctly in place.  If they aren't found it aborts the program."""
	existing_groups = params["security_groups"]
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
	if groups_found == len(existing_groups):
		logging.debug("All pre-existing groups found, continuing".format(group))
	if not new_group_found:
		logging.critical("Group {0} not found, aborting".format(params["name"]))
		#TODO: Throw group not found Exception
		sys.exit()
	for group in all_groups:
		logging.debug(group)
	params["all_security_groups"] = all_groups

def get_cluster_health(params):
	"""Poll for cluster status until all nodes are 'green'"""
	permanent_es_nodes = [params["permanent_instances"][0].private_ip_address]
	logging.debug("Polling for cluster status green")
	max_attempts, sleep_between_attempts = 30, 10
	#Try multiple times to get cluster health of green
	#We want to make sure to target all ec2_on_slave slave nodes plus the permanent master nodes
	target_nodes = len(params["ec2_on_slave"]) + len(params["permanent_instances"])
	status = "unknown"
	for j in range(0, max_attempts):
		try:
			if j > 0:
				logging.debug("Making attempt {0} of {1} for cluster status.".format(j,\
					max_attempts))
				try:
					es_connection = Elasticsearch(permanent_es_nodes, sniff_on_start=True,\
						sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)
				except Exception as err:
					logging.warning("Exception while trying to make Elasticsearch connection.")
					raise("Error trying to connect to ES node.")
				logging.debug("Cluster found.")
				logging.debug("Attempting to poll for health.")
				status, number_of_nodes = "unknown", 0
				while status != "green":
					result = es_connection.cluster.health()
					if result:
						status = result["status"]
						number_of_nodes = result["number_of_nodes"]
						#TODO: Return with vital information
						return status, number_of_nodes, target_nodes

						if status != "green":
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
	return status, number_of_nodes, target_node

def get_es_health(es_conn):
	"""Returns either a valid ElasticSearch health status or a warning."""
	es_health = None
	try:
		es_health = es_conn.cluster.health()
	except Exception as err:
		logging.warning("Exception while trying to pull Elasticsearch stats.")
	return es_health

def get_cluster_health_2(params):
	"""Collect highest CPU and highest search queue metrics from an ES cluster"""
	max_attempts, sleep_between_attempts = 30, 10
	#This pattern allows us to pull out the proper ip4 address
	#TODO: Remove 'target_nodes', this function doesn't do anything with that param!
	status, number_of_nodes, target_nodes = None, None, None
	for j in range(max_attempts):
		get_es_connection(params)
		#0. Terminate after too many attempts
		if j >= max_attempts:
			logging.critical("Error getting cluster statistics, aborting abnormally.")
			sys.exit()
		#1. Attempt to connect to ES cluster
		es_connection = get_es_connection(params)
		if not es_connection:
			j += 1
			continue
		#2. Attempt to connect to ES cluster
		es_health = get_es_health(es_connection)
		if not es_health:
			j += 1
			continue
		#3. Process the es_health
		status = es_health["status"]
		number_of_nodes = es_health["number_of_nodes"]
		target_nodes = len(params["ec2_on_slave"]) + len(params["permanent_instances"])
		return status, number_of_nodes, target_nodes

def get_cluster_metrics(params):
	"""Collect highest CPU and highest search queue metrics from an ES cluster"""
	max_attempts, sleep_between_attempts = 30, 10
	#This pattern allows us to pull out the proper ip4 address
	ip_address_pattern = re.compile("inet..(.*)......")
	for j in range(max_attempts):
		get_es_connection(params)
		#0. Terminate after too many attempts
		if j >= max_attempts:
			logging.critical("Error getting cluster statistics, aborting abnormally.")
			sys.exit()
		#1. Attempt to connect to ES cluster
		es_connection = get_es_connection(params)
		if not es_connection:
			j += 1
			continue
		#2. Attempt to connect to ES cluster
		es_stats = get_es_stats(es_connection)
		if not es_stats:
			j += 1
			continue
		#3. Process the es_stats
		nodes = es_stats["nodes"]
		#node_summaries = { "nodes": [] }
		highest_cpu, highest_queue = 0, 0
		#Loop through stats for each node in the cluster
		for n in nodes.keys():
			ip = ip_address_pattern.match(nodes[n]["ip"][0]).group(1)
			this_cpu = nodes[n]["os"]["cpu"]["usage"]
			search_queue = nodes[n]["thread_pool"]["search"]["queue"]
			if this_cpu > highest_cpu:
				highest_cpu = this_cpu
			if search_queue > highest_queue:
				highest_queue = search_queue
		#Collect a metrics of the highest cpu and the highest search queue
		return { "highest_cpu": highest_cpu, "highest_queue": highest_queue }

def get_es_connection(params):
	"""Returns either a valid ElasticSearch connection or a warning."""
	connection_nodes = [params["permanent_instances"][0].private_ip_address]
	es_connection = None
	try:
		es_connection = Elasticsearch(connection_nodes, sniff_on_start=True,\
			sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)
	except Exception as err:
		logging.warning("Exception while trying to make Elasticsearch connection.")
	return es_connection

def get_es_stats(es_conn):
	"""Returns either a valid ElasticSearch statistics or a warning."""
	es_stats = None
	try:
		es_stats = es_conn.nodes.stats(metric="name,os,thread_pool")
	except Exception as err:
		logging.warning("Exception while trying to pull Elasticsearch stats.")
	return es_stats

def get_parameters():
	"""Validates the command line arguments and loads a dict of params."""
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

def get_judgment(params, metrics):
	"""This function uses rules to get a judgment scaling up or down."""
	#Rules
	rules = params["scaling_rules"]
	#EC2 metrics
	ec2_on_slave, ec2_off_slave = len(params["ec2_on_slave"]), len(params["ec2_off_slave"])
	#ES metrics
	cpu, q = metrics["highest_cpu"], metrics["highest_queue"]

	#Lambda functions to create words for decision logic
	underloaded = lambda: cpu <= rules["down"]["cpu"] and q <= rules["down"]["search_queue"]
	overloaded = lambda: cpu >= rules["up"]["cpu"] and q >= rules["up"]["search_queue"]
	room_to_shrink = lambda: len(params["ec2_on_slave"]) >= params["nodes"]["minimum"]
	room_to_grow = lambda: len(params["ec2_off_slave"]) > 0

	#Make a judgment about whether to scale up, scale down, or do nothing
	judgment = None
	if underloaded:
		if room_to_shrink:
			judgment = "scale down"
		else:
			judgment = "neutral: scale down limit reached"
	elif overloaded:
		if room_to_grow:
			judgment = "scale up"
		else:
			judgment = "neutral: scale up limit reached"
	else:
		judgment = "neutral"
	logging.warning("IN-USE: {0}, STANDING-BY: {1}, CPU: {2}, QUEUE: {3}, RECOMMENDATION {4}".format(\
		ec2_on_slave, ec2_off_slave, cpu, q, judgment))
	return judgment

def pretty(my_json):
	"""A pretty print function"""
	return json.dumps(my_json, sort_keys=False, indent=4, separators=(',', ': '))

def refresh_ec2_metrics(params, ec2_conn):
	"""This function uses gathers ec2_metrics"""
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
	#Filtered instances not in master_instances
	instances = [ i for i in filtered_instances if i.id not in params["master_instances"] ]
	perm = [ i for i in filtered_instances if i.id in params["master_instances"] ]
	params["permanent_instances"] = perm
	#Number of ec2 nodes that are running
	params["ec2_on_slave"] = [i for i in instances if i.state == 'running' ]
	params["ec2_off_slave"] = [i for i in instances if i.state != 'running' ]
	params["instances"] = params["ec2_on_slave"]

def scale_down(params):
	"""Scales our elasticsearch cluster down."""
	logging.info("Scaling down.")
	ec2_on_slave = params["ec2_on_slave"]
	#TODO: switch to a random ec2_slave
	candidate = ec2_on_slave[0]
	logging.info("Stopping {0}".format(candidate.id))
	try:
		candidate.stop()
	except EC2ResponseError as err:
		logging.critical("Error scaling down, aborting: Exception {0}".format(err))
		logging.critical("Unexpected error:", sys.exc_info()[0])
		sys.exit()
	status, current_nodes = None, None
	#We need to achieve the number of ec2_on_slave instances plus the permanent instances
	target_nodes = len(params["ec2_on_slave"]) + len(params["permanent_instances"])
	#Since we are trying to reduce, reduce our target by one
	target_nodes += -1
	#Wait until the ES cluster is healthy (green) and has the correct number of nodes
	while (status != "green") or (current_nodes != target_nodes):
		status, current_nodes, _ = get_cluster_health_2(params)
		logging.warning("STATUS: {0}, CURRENT_NODES {1}, TARGET_NODES: {2}".format(status, current_nodes, target_nodes))
		time.sleep(5)

def scale_up(params):
	"""Scales our elasticsearch cluster up."""
	#TODO: Implement
	pass

def something_else(params):
	"""Sends an alert or something, I don't know yet."""
	#TODO" Implement
	pass

def start():
	"""This function starts the monitor."""
	params = get_parameters()
	my_region = boto.ec2.get_region(params["region"])
	ec2_conn = EC2Connection(region=my_region)
	confirm_security_groups(ec2_conn, params)
	interval = params["scaling_rules"]["interval"]
	consistency = 0
	previous_judgment, judgment = None, None

	#Continuously judge the status of the EC2 against available resources
	#and decide whether to scale up or down.  If there are a consistent
	#string of judgements to scale, then perform a scaling action.
	while True:
		refresh_ec2_metrics(params, ec2_conn)
		metrics = get_cluster_metrics(params)
		judgment = get_judgment(params, metrics)
		# Consistent judgments will trigger an action
		if judgment == "neutral":
			consistency = 0
		elif previous_judgment == judgment:
			consistency += 1
		else:
			consistency = 0
		if consistency >= params["scaling_rules"]["limit_break"]:
			logging.warning("Time to {0}".format(judgment))
			if judgment == "scale down":
				scale_down(params)
				consistency = 0
			elif judgment == "scale up":
				scale_up(params)
				consistency = 0
			else:
				something_else(params)
				consistency = 0
		previous_judgment = judgment
		time.sleep(interval)

if __name__ == "__main__":
	#MAIN PROGRAM
	logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/auto_scaler.log', \
		level=logging.INFO)
	logging.info("Auto_scale module activated.")
	start()

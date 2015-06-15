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
import json
import logging
import re
import sys
import time

from elasticsearch import Elasticsearch
from boto.ec2.connection import EC2Connection
from boto.exception import EC2ResponseError

from .custom_exceptions import FileProblem, InvalidArguments, UnknownJudgment, SecurityGroupNotFound

def confirm_security_groups(conn, params):
	"""Confirms that the security groups we need for accessng our cluster
	are correctly in place.  If those groups are not found the program
	aborts."""
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
		logging.debug("All pre-existing groups found, continuing")
	if not new_group_found:
		logging.critical("Supply the following arguments: config_file, cluster-name")
		raise SecurityGroupNotFound(msg="Cannot proceed without a valid Security Group", expr=None)
	for group in all_groups:
		logging.debug(group)
	params["all_security_groups"] = all_groups

def get_cluster_health(params):
	"""Collect highest CPU and highest search queue metrics from an ES cluster"""
	max_attempts = 30
	#This pattern allows us to pull out the proper ip4 address
	status, number_of_nodes = None, None
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
		#2. Attempt to pull cluster health
		es_health = get_es_health(es_connection)
		if not es_health:
			j += 1
			continue
		#3. Process the es_health
		status = es_health["status"]
		number_of_nodes = es_health["number_of_nodes"]
		# target_nodes = len(params["ec2_on_slave"]) + len(params["permanent_instances"])
		return status, number_of_nodes

def get_cluster_metrics(params):
	"""Collect highest CPU and highest search queue metrics from an ES cluster"""
	max_attempts = 30
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
		#2. Attempt to pull cluster stats
		es_stats = get_es_stats(es_connection)
		if not es_stats:
			j += 1
			continue
		#3. Process the es_stats
		nodes = es_stats["nodes"]
		#node_summaries = { "nodes": [] }
		highest_cpu, highest_queue = 0, 0
		#Loop through stats for each node in the cluster
		for key in nodes.keys():
			# ip = ip_address_pattern.match(nodes[key]["ip"][0]).group(1)
			this_cpu = nodes[key]["os"]["cpu"]["usage"]
			search_queue = nodes[key]["thread_pool"]["search"]["queue"]
			if this_cpu > highest_cpu:
				highest_cpu = this_cpu
			if search_queue > highest_queue:
				highest_queue = search_queue
		#Collect a metrics of the highest cpu and the highest search queue
		return {"highest_cpu": highest_cpu, "highest_queue": highest_queue}

def get_es_connection(params):
	"""Returns either a valid ElasticSearch connection or a warning."""
	connection_nodes = [params["permanent_instances"][0].private_ip_address]
	es_connection = None
	try:
		es_connection = Elasticsearch(connection_nodes, sniff_on_start=True,\
			sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)
	except Exception:
		logging.warning("Exception while trying to make Elasticsearch connection.")
	return es_connection

def get_es_health(es_conn):
	"""Returns either a valid ElasticSearch health status or a warning."""
	es_health = None
	try:
		es_health = es_conn.cluster.health()
	except Exception:
		logging.warning("Exception while trying to pull Elasticsearch stats.")
	return es_health

def get_es_stats(es_conn):
	"""Returns either a valid ElasticSearch statistics or a warning."""
	es_stats = None
	try:
		es_stats = es_conn.nodes.stats(metric="name,os,thread_pool")
	except Exception:
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
		logging.critical("%s not found, aborting.", sys.argv[1])
		raise FileProblem(msg="Cannot find a valid configuration file.", expr=None)
	return params

def get_judgment(params, metrics):
	"""This function uses rules to get a judgment scaling up or down."""
	#Rules
	rules = params["scaling_rules"]
	#EC2 metrics
	ec2_on_slave, ec2_off_slave = len(params["ec2_on_slave"]), len(params["ec2_off_slave"])
	#ES metrics
	cpu, queue = metrics["highest_cpu"], metrics["highest_queue"]

	#Lambda functions to create words for decision logic
	underloaded = lambda: (cpu <= rules["down"]["cpu"]) and (queue <= rules["down"]["search_queue"])
	overloaded = lambda: (cpu >= rules["up"]["cpu"]) and (queue >= rules["up"]["search_queue"])
	room_to_shrink = lambda: len(params["ec2_on_slave"]) >= params["nodes"]["minimum"]
	room_to_grow = lambda: len(params["ec2_off_slave"]) > 0

	#Make a judgment about whether to scale up, scale down, or do nothing
	judgment = None
	
	if underloaded():
		if room_to_shrink():
			judgment = "contract"
		else:
			judgment = "neutral: scale down limit reached"
	elif overloaded():
		if room_to_grow():
			judgment = "expand"
		else:
			judgment = "neutral: scale up limit reached"
	else:
		judgment = "neutral"
	logging.warning("IN-USE: {0}, \
		             STANDING-BY: {1}, \
		             CPU: {2}, \
		             QUEUE: {3}, \
		             RECOMMENDATION {4}".format(\
		ec2_on_slave, ec2_off_slave, cpu, queue, judgment))
	return judgment

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
	instances = [i for i in filtered_instances if i.id not in params["master_instances"]]
	perm = [i for i in filtered_instances if i.id in params["master_instances"]]
	params["permanent_instances"] = perm
	#Number of ec2 nodes that are running
	params["ec2_on_slave"] = [i for i in instances if i.state == 'running']
	params["ec2_off_slave"] = [i for i in instances if i.state != 'running']
	params["instances"] = params["ec2_on_slave"]

def wait_for_healthy_cluster(params, target_nodes):
	"""This module polls the cluster for its health and returns once the cluster
	is found to be in good health."""
	status, current_nodes = None, None
	while (status != "green") or (current_nodes != target_nodes):
		status, current_nodes = get_cluster_health(params)
		logging.warning("STATUS: {0}, \
			             CURRENT_NODES {1}, \
			             TARGET_NODES: {2}".format( \
			             	status, \
			             	current_nodes, \
			             	target_nodes))
		time.sleep(5)

def scale(params, judgment):
	"""Scales our elasticsearch cluster up or down."""
	logging.info("Scale action is: {0}.".format(judgment))
	#TODO: switch to a random ec2_slave candidate
	candidate, target_offset = None, None
	try:
		if judgment == "contract":
			candidate = params["ec2_on_slave"][0]
			candidate.stop()
			target_offset = -1
		elif judgment == "expand":
			candidate = params["ec2_off_slave"][0]
			candidate.start()
			target_offset = 1
		else:
			raise UnknownJudgment(msg="No handling for that judgment, aborting.", \
				expr=None)
		logging.info("Candidate is {0}".format(candidate.id))
	except EC2ResponseError as err:
		logging.critical("Error scaling down, aborting: Exception {0}".format(err))
		logging.critical("Unexpected error: "+ str(sys.exc_info()[0]))
		sys.exit()
	# We need to achieve the number of ec2_on_slave instances
	# plus the permanent instances
	target_nodes = len(params["ec2_on_slave"]) + len(params["permanent_instances"])
	# Since we are trying to reduce, reduce our target by one
	target_nodes += target_offset
	# Wait until the ES cluster is healthy (green)
	# and has the correct number of nodes
	wait_for_healthy_cluster(params, target_nodes)

def something_else():
	"""Sends an alert or something, I don't know yet."""
	#TODO: Implement
	logging.critical("Unimplemented, 'something_else' function.")

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
		if consistency >= params["scaling_rules"]["consensus_limit"]:
			logging.warning("Time to {0}".format(judgment))
			if judgment in ["contract", "expand"]:
				scale(params, judgment)
				consistency = 0
			else:
				something_else()
				consistency = 0
		previous_judgment = judgment
		time.sleep(interval)

if __name__ == "__main__":
	#MAIN PROGRAM
	logging.basicConfig(format='%(asctime)s %(message)s', \
		filename='logs/auto_scaler.log', \
		level=logging.INFO)
	logging.info("Auto_scale module activated.")
	start()

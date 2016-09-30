"""This module updetes all dictionaries from a local host to a remote host, if the timestamps
require it.."""
import logging
import sys
import os
import re
import json
import yaml
import paramiko

from plumbum import SshMachine, local
from plumbum.commands.processes import ProcessExecutionError

from meerkat.geomancer.get_merchant_dictionaries import merge
from meerkat.various_tools import load_params
from meerkat.geomancer.pybossa.build_pybossa_project import format_json_with_callback

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
LOGGER = logging.getLogger('update_dictionaries')

def local_file_upload(args, s_file, d_file):
	"""Uploads files using sftp."""
	try:
		transporter = paramiko.Transport((args["ip_address"], args["port"]))
		key_path = os.path.expanduser(args["key_path"])
		key = paramiko.RSAKey.from_private_key_file(key_path)
		transporter.connect(username=args["username"], pkey=key)
		sftp = paramiko.SFTPClient.from_transport(transporter)
		_ = sftp.put(s_file, d_file)
	except Exception as my_exception:
		print("LocalScript inited Failed", my_exception)
		return False
	else:
		transporter.close()

def get_timestamp_of_remote_file(file_path):
	"""Get the timestamp of a remote file."""
	my_pattern = re.compile(r'.*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\S{13}).*')
	with SshMachine("52.37.199.197", user="ubuntu", keyfile="/home/ubuntu/.ssh/twu.pem") as rem:
		timestamp = rem["ls"]["-Fal"]["--time-style=full-iso"][file_path]
		try:
			stdout = timestamp()
		except ProcessExecutionError as process_execution_error:
			LOGGER.critical("Remote file not found at: {0}, Error is {1}".format(file_path,
				process_execution_error))
			sys.exit()
		if my_pattern.search(stdout):
			matches = my_pattern.match(stdout)
			LOGGER.info("Timestamp of remote file {0}: {1}".format(file_path, matches.group(1)))
	return matches.group(1).strip()

def get_timestamp_of_local_file(file_path):
	"""Get the timestamp of a local file."""
	my_pattern = re.compile(r'.*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\S{13}).*')
	timestamp = local["ls"]["-Fal"]["--time-style=full-iso"][file_path]
	stdout = timestamp()
	if my_pattern.search(stdout):
		matches = my_pattern.match(stdout)
		LOGGER.info("Timestamp of local file {0}: {1}".format(file_path, matches.group(1)))
	else:
		LOGGER.info("No match")
	return matches.group(1).strip()

def real_main():
	"""This module does the majority of the work."""
	merchant = "AutoZone" #FIXME Shouldn't this be a parameter??

	remote_prefix = "/var/www/html/dictionaries/"
	remote_dict_path = remote_prefix + merchant + "/geo.json"
	remote_timestamp = get_timestamp_of_remote_file(remote_dict_path)

	local_prefix = "meerkat/geomancer/merchants/"
	agg_dict_path = local_prefix + merchant + "/geo.json"
	if os.path.isfile(agg_dict_path):
		agg_timestamp = get_timestamp_of_local_file(agg_dict_path)
	else:
		LOGGER.critical("Agg dictionary doesn't exist, aborting!")
		sys.exit()

	agg_dict = load_params(agg_dict_path)
	geomancer_dict_path = local_prefix + merchant + "/geomancer.json"
	if os.path.isfile(geomancer_dict_path):
		geomancer_timestamp = get_timestamp_of_local_file(geomancer_dict_path)
		if geomancer_timestamp > remote_timestamp or agg_timestamp > remote_timestamp:
			LOGGER.info("Merging agg_dict onto geomancer_dict")
			geomancer_dict = load_params(geomancer_dict_path)
			merge(geomancer_dict, agg_dict)
			LOGGER.info("Uploading geomancer_dict")
			updated_dict_path = local_prefix + merchant + "/updated.json"
			with open(updated_dict_path, "w") as outfile:
				json.dump(geomancer_dict, outfile)

			format_json_with_callback(updated_dict_path, 0)
			upload_args = {
				"ip_address": "52.37.199.197",
				"port" : 22,
				"username": "ubuntu",
				"key_path": "/home/ubuntu/.ssh/twu.pem"
			}
			local_file_upload(upload_args, updated_dict_path, remote_dict_path)

	elif agg_timestamp > remote_timestamp:
		LOGGER.info("Uploading agg_dict")
		updated_dict_path = local_prefix + merchant + "/updated.json"
		with open(updated_dict_path, "w") as outfile:
			json.dump(agg_dict, outfile)
		format_json_with_callback(updated_dict_path, 0)
	else:
		LOGGER.info("No updates needed.")

	LOGGER.info(remote_timestamp)
	LOGGER.info(agg_timestamp)
	LOGGER.info(geomancer_timestamp)
	LOGGER.info(remote_timestamp > agg_timestamp)
	LOGGER.info(remote_timestamp > geomancer_timestamp)

if __name__ == "__main__":
	real_main()



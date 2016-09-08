import logging
import yaml
import sys
import os
import re
import json
import paramiko

from meerkat.geomancer.get_merchant_dictionaries import merge
from meerkat.various_tools import load_params
from meerkat.geomancer.pybossa.build_pybossa_project import format_json_with_callback
from plumbum import SshMachine
from plumbum import local
from plumbum.commands.processes import ProcessExecutionError

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('update_dictionaries')

def local_file_upload(ip, port, username, key_path, s_file, d_file):
	try:
		t = paramiko.Transport((ip,port))
		KeyPath=os.path.expanduser(key_path)
		key=paramiko.RSAKey.from_private_key_file(KeyPath)
		t.connect(username = username,pkey=key)
		sftp = paramiko.SFTPClient.from_transport(t)
		ret=sftp.put(s_file,d_file)
	except Exception as e:
		print("LocalScript inited Failed",e)
		return False
	else:
		t.close()

def get_timestamp_of_remote_file(file_path):
	my_pattern = re.compile(r'.*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\S{13}).*')
	with SshMachine("52.37.199.197", user="ubuntu", keyfile="/home/ubuntu/.ssh/twu.pem") as rem:
		timestamp = rem["ls"]["-Fal"]["--time-style=full-iso"][file_path]
		try:
			stdout = timestamp()
		except ProcessExecutionError as e:
			logger.critical("Remote file not found at: {0}".format(file_path))
			sys.exit()
		if my_pattern.search(stdout):
			matches = my_pattern.match(stdout)
			logger.info("Timestamp of remote file {0}: {1}".format(file_path, matches.group(1)))
	return matches.group(1).strip()

def get_timestamp_of_local_file(file_path):
	my_pattern = re.compile(r'.*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\S{13}).*')
	timestamp = local["ls"]["-Fal"]["--time-style=full-iso"][file_path]
	stdout = timestamp()
	if my_pattern.search(stdout):
		matches = my_pattern.match(stdout)
		logger.info("Timestamp of local file {0}: {1}".format(file_path, matches.group(1)))
	else:
		logger.info("No match")
	return matches.group(1).strip()

merchant = "AutoZone"

remote_prefix = "/var/www/html/dictionaries/"
remote_dict_path = remote_prefix + merchant + "/geo.json"
remote_timestamp = get_timestamp_of_remote_file(remote_dict_path)

local_prefix = "meerkat/geomancer/merchants/"
agg_dict_path = local_prefix + merchant + "/geo.json"
if os.path.isfile(agg_dict_path):
	agg_timestamp = get_timestamp_of_local_file(agg_dict_path)
else:
	logger.critical("Agg dictionary doesn't exist, aborting!")
	sys.exit()

agg_dict = load_params(agg_dict_path)
geomancer_dict_path = local_prefix + merchant + "/geomancer.json"
if os.path.isfile(geomancer_dict_path):
	geomancer_timestamp = get_timestamp_of_local_file(geomancer_dict_path)
	if geomancer_timestamp > remote_timestamp or agg_timestamp > remote_timestamp:
		logger.info("Merging agg_dict onto geomancer_dict")
		geomancer_dict = load_params(geomancer_dict_path)
		merge(geomancer_dict, agg_dict)
		logger.info("Uploading geomancer_dict")
		#FIXME upload_to_remote(geomancer_dict)
		updated_dict_path = local_prefix + merchant + "/updated.json"
		with open(updated_dict_path, "w") as outfile:
			json.dump(geomancer_dict, outfile)

		format_json_with_callback(updated_dict_path, 0)
		ip, port = "52.37.199.197", 22
		username, key_path = "ubuntu", "/home/ubuntu/.ssh/twu.pem"
		local_file_upload(ip, port, username, key_path, updated_dict_path, remote_dict_path)

elif agg_timestamp > remote_timestamp:
		logger.info("Uploading agg_dict")
		#FIXME upload_to_remote(agg_dict)
		updated_dict_path = local_prefix + merchant + "/updated.json"
		with open(updated_dict_path, "w") as outfile:
			json.dump(agg_dict, outfile)
		format_json_with_callback(updated_dict_path, 0)
else:
	logger.info("No updates needed.")




logger.info(remote_timestamp)
logger.info(agg_timestamp)
logger.info(geomancer_timestamp)
logger.info(remote_timestamp > agg_timestamp)
logger.info(remote_timestamp > geomancer_timestamp)

#if remote_timestamp > agg_timestamp and remote_timestamp > geomancer_timestamp
"""
other_dict = load_params("dummy2.json")
return_dict = load_params("dummy.json")

logger.info("Return_dict {0}".format(return_dict))
logger.info("Other_dict {0}".format(other_dict))
logger.info("Blending")
logger.info("Blending")
merge(return_dict, other_dict)
logger.info("Return_dict {0}".format(return_dict))
"""


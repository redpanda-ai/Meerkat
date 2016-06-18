"""This module will update Meerkat's models from S3"""

import argparse
import json
import logging
import re
import tarfile
import os
import sys
import threading
from queue import Queue
import pandas as pd

from boto.s3 import connect_to_region
from boto.s3.key import Key
import boto3

from meerkat.various_tools import (safely_remove_file, validate_configuration
	, load_params, push_file_to_s3)

S3_CLIENT = boto3.client('s3')

def s3_key_exists(bucket, key):
	"""Determine whether object in S3 exists"""
	client = boto3.client('s3')
	results = client.list_objects(Bucket=bucket.name, Prefix=key)
	return 'Contents' in results

def get_peer_models(candidate_dictionary, prefix=None):
	"""Show the candidate models for each peer"""
	results = {}
	my_pattern = re.compile("(" + prefix + r")(.*/)(\d{14}/)")
	for key in candidate_dictionary:
		if my_pattern.search(key):
			matches = my_pattern.match(key)
			model_type, timestamp = matches.group(2), matches.group(3)
			if model_type not in results:
				results[model_type] = []
			results[model_type].append(timestamp)
			results[model_type].sort()
		else:
			logging.warning("Not Found")
	return results

def get_model_accuracy(classification_report):
	"""Retrieve the model accuracy from a correctly formed classification report."""
	df = pd.read_csv(classification_report)
	raw = str(df.iat[0, 0])
	if re.match(r"^\d+?\.\d+?$", raw) is None:
		logging.warning("Classification Report malformed, skipping this file.")
		return 0.0
	return float(raw)

def get_single_file_from_tarball(archive_name, archive, filename_pattern, extract=True):
	"""Untars and gunzips the stats file from the archive file"""
	if not tarfile.is_tarfile(archive):
		raise Exception("Invalid, not a tarfile.")
	my_pattern = re.compile(filename_pattern)
	with tarfile.open(name=archive, mode="r:gz") as tar:
		members = tar.getmembers()
		logging.debug("Members {0}".format(members))
		file_list = [member for member in members if my_pattern.search(member.name)]
		if len(file_list) != 1:
			format_string = "Archive {0} does not contain exactly one file matching pattern: {1}."
			logging.warning(format_string.format(archive_name, filename_pattern))
			raise Exception("Bad archive")
		else:
			my_file = file_list.pop()
			my_name = my_file.name
			if extract:
				tar.extract(my_file)
	return my_name

def fetch_tarball_and_extract(timestamp, target, **kwargs):
	"""Fetches a tarball from S3, pulls two csv files and uploads them back to S3"""
	k = Key(kwargs["bucket"])
	k.key = kwargs["prefix"] + kwargs["key"] + timestamp + target
	k.get_contents_to_filename(target)
	# Require Meta
	# Fixme, pretty sure we can just check the tarball for the meta file instead of extracting
	try:
		meta = get_single_file_from_tarball(timestamp, target, ".*meta")
		safely_remove_file(meta)
	except Exception:
		pass
	_ = get_single_file_from_tarball(timestamp, target, "confusion_matrix.csv")
	classification_report = get_single_file_from_tarball(timestamp, target,
		"classification_report.csv")
	logging.info("Tarball fetched and classification_report.csv extracted.")
	upload_path = kwargs["prefix"] + kwargs["key"] + timestamp
	push_file_to_s3("confusion_matrix.csv", kwargs["bucket"], upload_path)
	push_file_to_s3("classification_report.csv", kwargs["bucket"], upload_path)
	logging.info("Classification_report.csv pushed to S3.")
	return classification_report

def get_best_model_of_class(target, **kwargs):
	"""Finds the best candidate model of all contenders."""
	highest_score, candidate_count = 0.0, 1
	winner_count = candidate_count

	timestamps = None
	if kwargs["aspirant"] is None:
		timestamps = kwargs["results"][kwargs["key"]]
	else:
		timestamps = [kwargs["aspirant"]]

	for timestamp in timestamps:
		short_key = kwargs["prefix"] + kwargs["key"] + timestamp
		long_key = short_key + "classification_report.csv"
		classification_report = None
		if s3_key_exists(kwargs["bucket"], long_key):
			target_file = "classification_report.csv"
			S3_CLIENT.download_file(kwargs['bucket'].name, long_key, target_file)
			classification_report = target_file
			logging.info("Classification Report fetched from S3 at {0}.".format(long_key))
		else:
			logging.critical("Didn't find {0}".format(long_key))
			classification_report = fetch_tarball_and_extract(timestamp, target, **kwargs)
			logging.critical("Tarball fetched from S3 and classification_report extracted.")
		score = get_model_accuracy(classification_report)
		if score == 0.0:
			files_to_nuke = ["results.tar.gz", "classification_report.csv", "confusion_matrix.csv"]
			for item in files_to_nuke:
				logging.info("Removing {0} from {1}/{2}".format(item,
					kwargs["bucket"].name, short_key))
				S3_CLIENT.delete_object(Bucket=kwargs["bucket"].name, Key=short_key + item)

		logging.info("Score :{0}".format(score))

		if score > highest_score:
			highest_score = score
			kwargs["best_models"][kwargs["key"]] = timestamp
			winner_count = candidate_count
		logging.info("\t{0:<14}{1:>2}: {2:16}, Score: {3:0.5f}".format("Candidate",
			candidate_count, timestamp, score))
		candidate_count += 1
	return winner_count

def threaded_thing(**kwargs):
	"""This will be threaded"""
	# Ignore category models for now
	if "category" in kwargs["key"]:
		return

	logging.info("Evaluating '{0}'".format(kwargs["key"]))
	kwargs["aspirant"] = None
	if kwargs["key"] in kwargs["aspirants"]:
		logging.info("Aspirant model for {0} is {1}".format(kwargs["key"],
			kwargs["aspirants"][kwargs["key"]]))
		kwargs["aspirant"] = kwargs["aspirants"][kwargs["key"]]
	else:
		logging.info("No aspirant model, evaluating based on accuracy")

	target = kwargs["target"]
	del kwargs["target"]
	winner_count = get_best_model_of_class(target, **kwargs)

	logging.info("\t{0:<14}{1:>2}".format("Winner", winner_count))

def worker(**kwargs):
	"""Does work"""
	while True:
		_ = kwargs["q"].get()
		threaded_thing(**kwargs)
		kwargs["q"].task_done()

def get_best_models(*args):
	"""Gets the best model for a particular model type."""
	best_models = {}
	#make threads here
	#make an empty queue
	my_queue = Queue()
	kwargs = {
		"q": my_queue,
		"bucket": args[0],
		"prefix": args[1],
		"results": args[2],
		"target": args[3],
		"s3_base": args[4],
		"aspirants": args[5],
		"best_models": best_models
	}
	#start your threads
	sorted_keys = sorted(kwargs["results"].keys())
	logging.critical("Sorted keys: {0}".format(sorted_keys))
	for i, _ in enumerate(sorted_keys):
		my_kwargs = {}
		for key in kwargs:
			my_kwargs[key] = kwargs[key]

		my_kwargs["key"] = sorted_keys[i]
		my_thread = threading.Thread(target=worker, kwargs=my_kwargs)
		my_thread.daemon = True
		my_thread.start()
	#fill your queue
	for key in sorted_keys:
		logging.critical("Key placed")
		my_queue.put(key)
	#block until all tasks are done
	my_queue.join()

	#log the winners
	logging.debug("The best models are:\n{0}".format(best_models))
	load_winning_models(**kwargs)

def load_winning_models(**kwargs):
	"""Load the best models locally"""
	#Load the winners
	best_models, bucket = kwargs["best_models"], kwargs["bucket"].name
	prefix, target = kwargs["prefix"], kwargs["target"]
	etags, etags_file = get_etags()
	for key in sorted(best_models.keys()):
		timestamp = best_models[key]
		logging.info("Loading {0} for {1}".format(timestamp, key))
		load_best_model_for_type(bucket=bucket, model_type=key,
			timestamp=best_models[key], s3_prefix=prefix, etags=etags)

	with open(etags_file, "w") as outfile:
		logging.info("Writing {0}".format(etags_file))
		json.dump(etags, outfile)
	
		#Get the results.tar.gz file from the path in S3
		#Get the meta and json from the tarball and move to the correct path locally.
	# Cleanup
	safely_remove_file("confusion_matrix.csv")
	safely_remove_file(target)


def get_etags():
	"""Fetches local ETag values from a local file."""
	etags = {}
	etags_file = "meerkat/classification/etags.json"
	if os.path.isfile(etags_file):
		logging.info("ETags found.")
		etags = load_params(etags_file)
	else:
		logging.info("Etags not found, all models will be downloaded.")
	return etags, etags_file

def load_best_model_for_type(**kwargs):
	"""Loads the best model for a given model type from S3 to the local host."""
	client = boto3.client('s3')

	model_type = kwargs["model_type"]
	if model_type.startswith("/"):
		model_type = model_type[1:]

	if kwargs["s3_prefix"].endswith("/"):
		kwargs["s3_prefix"] = kwargs["s3_prefix"][:-1]
	remote_file = kwargs["s3_prefix"] + "/" + model_type +\
		kwargs["timestamp"] + "results.tar.gz"
	logging.debug("Bucket name {0}".format(kwargs["bucket"]))
	logging.debug("Remote file is {0}".format(remote_file))

	remote_etag = client.list_objects(Bucket=kwargs["bucket"],
		Prefix=remote_file)["Contents"][0]["ETag"]
	local_etag = None
	if remote_file in kwargs["etags"]:
		local_etag = kwargs["etags"][remote_file]

	logging.debug("{0: <6} ETag is : {1}".format("Remote", remote_etag))
	logging.debug("{0: <6} ETag is : {1}".format("Local", local_etag))

	#If the file is already local, skip downloading
	if local_etag == remote_etag:
		logging.info("Model exists locally no need to download")
		return

	client.download_file(kwargs["bucket"], remote_file, "results.tar.gz")
	if not tarfile.is_tarfile("results.tar.gz"):
		logging.critical("Tarball is invalid, aborting")
		sys.exit()
	logging.info("Tarball is valid, continuing")
	#Extract everything
	necessary_files = ["train.ckpt", "train.meta", "label_map.json", "classification_report.csv"]
	with tarfile.open(name="results.tar.gz", mode="r:gz") as tar:
		members = tar.getmembers()
		member_names = [member.name for member in members]
		for name in member_names:
			logging.debug("Member is {0}".format(name))
		for needed_file in necessary_files:
			if needed_file not in member_names:
				logging.critical("Archive does not contain {0}, aborting".format(needed_file))
				files_to_nuke = ["results.tar.gz", "classification_report.csv", "confusion_matrix.csv"]
				remote_base = kwargs["s3_prefix"] + "/" + model_type + kwargs["timestamp"]
				for item in files_to_nuke:
					logging.info("Removing {0} from {1}/{2}".format(item,
						kwargs["bucket"], remote_base))
					client.delete_object(Bucket=kwargs["bucket"], Key=remote_base + item)
				sys.exit()
		tar.extractall()
	logging.info("Tarball contents extracted.")
	output_path = "meerkat/classification/"
	#Move label_map
	new_path = output_path + "label_maps/" + get_asset_name(model_type, "", "json")
	logging.info("Moving label_map to: {0}".format(new_path))
	os.rename("label_map.json", new_path)
	#Move graph
	new_path = output_path + "models/" + get_asset_name(model_type, "", "meta")
	logging.info("Moving graph to: {0}".format(new_path))
	os.rename("train.meta", new_path)
	#Move checkpoint
	new_path = output_path + "models/" + get_asset_name(model_type, "", "ckpt")
	logging.info("Moving model to: {0}".format(new_path))
	os.rename("train.ckpt", new_path)

	kwargs["etags"][remote_file] = remote_etag

def get_asset_name(suffix, key, extension):
	"""Cleans up the name of a meta, json, or ckpt file, which may
	have a leading dot '.'"""
	result = (suffix + key).replace("/", ".") + extension
	if result.startswith("."):
		result = result[1:]
	return result

def find_s3_objects(s3_client=None, bucket=None, prefix=None, target=None):
	"""Doesn't use recursion, but finds all objects simply."""
	simple_results = {}
	object_names = s3_client.list_objects(Bucket=bucket, Prefix=prefix)["Contents"]
	for s3_object in object_names:
		key = s3_object["Key"]
		if key.endswith("/" + target):
			simple_results[key] = target
	return simple_results

def main_program(bucket="s3yodlee", region="us-west-2",
	prefix="meerkat/cnn/data", config=None):
	"""Execute the main program"""
	parser = argparse.ArgumentParser("auto_load")
	parser.add_argument("-l", "--log_level", default="warning",
		help="Show at least this level of logs")
	parser.add_argument("-b", "--bucket", default=bucket,
		help="Name of S3 bucket containing the candidate models.")
	parser.add_argument("-r", "--region", default=region,
		help="Name of the AWS region containing the S3 bucket")
	parser.add_argument("-p", "--prefix", default=prefix,
		help="S3 object prefix that precedes all object keys for our candidate models")
	parser.add_argument("-c", "--config", default=config,
		help="The local path to a JSON file of models have been pre-selected")
	args = parser.parse_args()
	log_format = "%(asctime)s %(levelname)s: %(message)s"
	if args.log_level == "debug":
		logging.basicConfig(format=log_format, level=logging.DEBUG)
	elif args.log_level == "info":
		logging.basicConfig(format=log_format, level=logging.INFO)
	else:
		logging.basicConfig(format=log_format, level=logging.INFO)
	aspirants = {}
	if args.config is not None:
		config = validate_configuration(args.config,
			"meerkat/classification/config/auto_load_schema.json")
		aspirants = config.get("aspirants", {})
		args.region = config.get("region", args.region)
		args.prefix = config.get("prefix", args.prefix)
		args.bucket = config.get("bucket", args.bucket)
	logging.warning("Aspirants are: {0}".format(aspirants))
	logging.warning("Region: {0}".format(args.region))
	logging.warning("Prefix: {0}".format(args.prefix))
	logging.warning("Bucket: {0}".format(args.bucket))

	logging.warning("Starting main program")
	conn = connect_to_region(args.region)
	bucket = conn.get_bucket(args.bucket)
	target = "results.tar.gz"

	my_results = find_s3_objects(s3_client=S3_CLIENT, bucket=args.bucket,
		prefix=args.prefix, target="results.tar.gz")

	results = get_peer_models(my_results, prefix=args.prefix)
	logging.debug("Results: {0}".format(results))
	get_best_models(bucket, args.prefix, results, target, args.prefix, aspirants)
	logging.warning("Finishing main program")

if __name__ == "__main__":
	#Execute the main program
	main_program()

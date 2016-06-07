"""This module will update Meerkat's models from S3"""

import argparse
import logging
import re
import tarfile
import os
import sys
import pandas as pd

from boto.s3 import connect_to_region
from boto.s3.key import Key
import boto3

from meerkat.various_tools import safely_remove_file, validate_configuration
from meerkat.classification.tools import push_file_to_s3

def find_s3_objects_recursively(conn, bucket, my_results, prefix=None, target=None):
	"""Find all S3 target objects and their locations recursively"""
	folders = bucket.list(prefix=prefix, delimiter="/")
	for s3_object in folders:
		if s3_object.name != prefix:
			last_slash = s3_object.name[-len(target) - 1] == "/"
			if s3_object.name[-len(target):] == target and last_slash:
				my_results[prefix] = target
				logging.debug("name is {0}".format(s3_object.name))
				return s3_object.name
			elif s3_object.name[-1:] == "/":
				find_s3_objects_recursively(conn, bucket, my_results, prefix=s3_object.name,
					target=target)

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
	except:
		pass
	_ = get_single_file_from_tarball(timestamp, target, "confusion_matrix.csv")
	classification_report = get_single_file_from_tarball(timestamp, target,
		"classification_report.csv")
	logging.warning("Tarball fetched and classification_report.csv extracted.")
	upload_path = kwargs["prefix"] + kwargs["key"] + timestamp
	push_file_to_s3("confusion_matrix.csv", kwargs["bucket"], upload_path)
	push_file_to_s3("classification_report.csv", kwargs["bucket"], upload_path)
	logging.warning("Classification_report.csv pushed to S3.")
	return classification_report

def get_best_model_of_class(target, **kwargs):
	"""Finds the best candidate model of all contenders."""
	highest_score, winner, candidate_count = 0.0, None, 1
	winner_count = candidate_count

	timestamps = None
	if kwargs["aspirant"] is None:
		timestamps = kwargs["results"][kwargs["key"]]
	else:
		timestamps = [kwargs["aspirant"]]

	s3_client = boto3.client('s3')
	for timestamp in timestamps:
		short_key = kwargs["prefix"] + kwargs["key"] + timestamp
		long_key = short_key + "classification_report.csv"
		classification_report = None
		if s3_key_exists(kwargs["bucket"], long_key):
			target_file = "classification_report.csv"
			s3_client.download_file(kwargs['bucket'].name, long_key, target_file)
			classification_report = target_file
			logging.critical("Classification Report fetched from S3 at {0}.".format(long_key))
		else:
			logging.critical("Didn't find {0}".format(long_key))
			classification_report = fetch_tarball_and_extract(timestamp, target, **kwargs)
			logging.critical("Tarball fetched from S3 and classification_report extracted.")
		score = get_model_accuracy(classification_report)
		if score == 0.0:
			files_to_nuke = ["results.tar.gz", "classification_report.csv", "confusion_matrix.csv"]
			for item in files_to_nuke:
				logging.warning("Removing {0} from {1}/{2}".format(item, kwargs["bucket"].name, short_key))
				s3_client.delete_object(Bucket=kwargs["bucket"].name, Key=short_key + item)

		logging.warning("Score :{0}".format(score))

		if score > highest_score:
			highest_score = score
			winner = timestamp
			kwargs["best_models"][kwargs["key"]] = timestamp
			winner_count = candidate_count
		logging.warning("\t{0:<14}{1:>2}: {2:16}, Score: {3:0.5f}".format("Candidate",
			candidate_count, timestamp, score))
		candidate_count += 1
	return winner_count, winner

def get_best_models(*args):
	"""Gets the best model for a particular model type."""
	bucket, prefix, results, target, s3_base, aspirants = args[:]
	suffix = prefix[len(s3_base):]

	best_models = {}
	for key in sorted(results.keys()):

		# Ignore category models for now
		if "category" in key:
			continue

		logging.warning("Evaluating '{0}'".format(key))
		aspirant = None
		if key in aspirants:
			logging.warning("Aspirant model for {0} is {1}".format(key, aspirants[key]))
			aspirant = aspirants[key]
		else:
			logging.warning("No aspirant model, evaluating based on accuracy")

		winner_count, winner = get_best_model_of_class(target, bucket=bucket,
			prefix=prefix, results=results, key=key, suffix=suffix, aspirant=aspirant,
			best_models=best_models)

		args = [bucket, prefix, key, winner, s3_base, "results.tar.gz", "meerkat/classification/"]
		logging.warning("\t{0:<14}{1:>2}".format("Winner", winner_count))

	#log the winners
	logging.warning("The best models are:\n{0}".format(best_models))
	#Process the winners
	for key in sorted(best_models.keys()):
		timestamp = best_models[key]
		logging.warning("Processing {0} for {1}".format(timestamp, key))
		load_best_model_for_type(bucket=bucket.name, model_type=key, timestamp=best_models[key],
			s3_prefix=prefix)
		#Get the results.tar.gz file from the path in S3
		#Get the meta and json from the tarball and move to the correct path locally.
	# Cleanup
	safely_remove_file("confusion_matrix.csv")
	safely_remove_file(target)

def load_best_model_for_type(**kwargs):
	"""Loads the best model for a given model type from S3 to the local host."""
	client = boto3.client('s3')
	#Note: we removing the leading "/" from the remote_file using the number 1
	model_type = kwargs["model_type"][1:]
	remote_file = kwargs["s3_prefix"] + "/" + model_type +\
		kwargs["timestamp"] + "results.tar.gz"
	logging.critical("Bucket name {0}".format(kwargs["bucket"]))
	logging.critical("Remote file is {0}".format(remote_file))
	client.download_file(kwargs["bucket"], remote_file, "results.tar.gz")
	if not tarfile.is_tarfile("results.tar.gz"):
		logging.critical("Tarball is invalid, aborting")
		sys.exit()
	logging.warning("Tarball is valid, continuing")
	#Extract everything
	with tarfile.open(name="results.tar.gz", mode="r:gz") as tar:
		tar.extractall()
	logging.warning("Tarball contents extracted.")
	output_path = "meerkat/classification/"
	#Move label_map
	new_path = output_path + "label_maps/" + get_asset_name(model_type, "", "json")
	logging.warning("Moving label_map to: {0}".format(new_path))
	os.rename("label_map.json", new_path)
	#Move graph
	new_path = output_path + "models/" + get_asset_name(model_type, "", "meta")
	logging.warning("Moving graph to: {0}".format(new_path))
	os.rename("train.meta", new_path)
	#Move checkpoint
	new_path = output_path + "models/" + get_asset_name(model_type, "", "ckpt")
	logging.warning("Moving model to: {0}".format(new_path))
	os.rename("train.ckpt", new_path)


def get_asset_name(suffix, key, extension):
	"""Cleans up the name of a meta, json, or ckpt file, which may
	have a leading dot '.'"""
	result = (suffix + key).replace("/", ".") + extension
	if result.startswith("."):
		result = result[1:]
	return result

def main_program(bucket="s3yodlee", region="us-west-2", prefix="meerkat/cnn/data",
	config=None):
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
	my_results, target = {}, "results.tar.gz",
	find_s3_objects_recursively(conn, bucket, my_results, prefix=args.prefix, target=target)
	results = get_peer_models(my_results, prefix=args.prefix)
	logging.debug("Results: {0}".format(results))
	get_best_models(bucket, args.prefix, results, target, args.prefix, aspirants)
	logging.warning("Finishing main program")

if __name__ == "__main__":
	#Execute the main program
	main_program()

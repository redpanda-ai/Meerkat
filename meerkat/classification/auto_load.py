"""This module will update Meerkat's models from S3"""

import argparse
import logging
import re
import tarfile
import os
import pandas as pd

from boto.s3 import connect_to_region
from boto.s3.key import Key

from meerkat.various_tools import safely_remove_file, validate_configuration

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

def get_peer_models(candidate_dictionary, prefix=None):
	"""Show the candidate models for each peer"""
	results = {}
	my_pattern = re.compile("(" + prefix + ")(.*/)(\d{14}/)")
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

def get_model_accuracy(confusion_matrix):
	"""Gets the accuracy for a particular confusion matrix"""
	df = pd.read_csv(confusion_matrix)
	rows, cols = df.shape
	logging.debug("Rows: {0} x Cols {1}".format(rows, cols))
	if cols == rows + 3:
		#Support the old style of confusion matrix, eliminate pointless columns
		df = df.drop(df.columns[[0, 1, -1]], axis=1)
		#Reset columns names, which are off by one
		df.rename(columns=lambda x: int(x)-1, inplace=True)

	true_positive = pd.DataFrame(df.iat[i, i] for i in range(rows))
	accuracy = true_positive.sum() / df.sum().sum()

	return accuracy.values[0]

def get_single_file_from_tarball(archive_name, archive, filename_pattern):
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
			tar.extract(my_file)
	return my_name

def get_best_model_of_class(target, models_dir, **kwargs):
	"""Finds the best candidate model of all contenders."""
	highest_score, winner, candidate_count = 0.0, None, 1
	winner_count = candidate_count
	#Start inner loop
	timestamps = None
	if kwargs["aspirant"] is None:
		timestamps = kwargs["results"][kwargs["key"]]
	else:
		timestamps = [kwargs["aspirant"]]

	for timestamp in timestamps:
		k = Key(kwargs["bucket"])
		k.key = kwargs["prefix"] + kwargs["key"] + timestamp + target
		k.get_contents_to_filename(target)
		# Require Meta
		try:
			meta = get_single_file_from_tarball(timestamp, target, ".*meta")
			safely_remove_file(meta)
		except:
			continue

		matrix = get_single_file_from_tarball(timestamp, target, "confusion_matrix.csv")
		score = get_model_accuracy(matrix)

		if score > highest_score:
			highest_score = score
			winner = timestamp
			winner_count = candidate_count
			leader_model = get_single_file_from_tarball(timestamp, target, ".*ckpt")
			new_model_path = models_dir + get_asset_name(kwargs["suffix"], kwargs["key"], "ckpt")
			os.rename(leader_model, new_model_path)
		logging.info("\t{0:<14}{1:>2}: {2:16}, Score: {3:0.5f}".format("Candidate",
			candidate_count, timestamp, score))
		candidate_count += 1
	return winner_count, winner
	#End inner loop

def get_best_models(bucket, prefix, results, target, s3_base, aspirants):
	"""Gets the best model for a particular model type."""

	suffix = prefix[len(s3_base):]
	models_dir = "meerkat/classification/models/"

	for key in sorted(results.keys()):

		# Ignore category models for now
		if "category" in key:
			continue

		logging.info("Evaluating '{0}'".format(key))
		aspirant = None
		if key in aspirants:
			logging.info("Aspirant model for {0} is {1}".format(key, aspirants[key]))
			aspirant = aspirants[key]
		else:
			logging.info("No aspirant model, evaluating based on accuracy")

		winner_count, winner = get_best_model_of_class(target, models_dir, bucket=bucket,
			prefix=prefix, results=results, key=key, suffix=suffix, aspirant=aspirant)

		args = [bucket, prefix, key, winner, s3_base, "results.tar.gz", "meerkat/classification/"]
		set_label_map_and_meta(*args)
		logging.info("\t{0:<14}{1:>2}".format("Winner", winner_count))

	# Cleanup
	safely_remove_file("confusion_matrix.csv")
	safely_remove_file(target)

def get_asset_name(suffix, key, extension):
	result = (suffix + key).replace("/", ".") + extension
	if result.startswith("."):
		result = result[1:]
	return result

def set_label_map_and_meta(*args):
	"""Moves the appropriate label map and meta files from a tarball in S3 to 
	a specific path on the local machine."""
	bucket, prefix, key, winner, s3_base, tarball, output_path = args[:]

	suffix = prefix[len(s3_base):]

	if bucket is not None: #None is used for unit tests
		s3_key = Key(bucket)
		s3_key.key = prefix + key + winner + tarball
		s3_key.get_contents_to_filename(tarball)

	#Move label_map
	json_file = get_single_file_from_tarball("", tarball, ".*json")
	new_path = output_path + "label_maps/" + get_asset_name(suffix, key, "json")
	logging.debug("Moving label_map to: {0}".format(new_path))
	os.rename(json_file, new_path)
	#Move graph
	meta_file = get_single_file_from_tarball("", tarball, ".*meta")
	new_graph_def_path = output_path + "models/" + get_asset_name(suffix, key, "meta")
	os.rename(meta_file, new_graph_def_path)
	return new_path

def main_program(bucket="s3yodlee", region="us-west-2", prefix="meerkat/cnn/data",
	config=None):
	"""Execute the main program"""
	parser = argparse.ArgumentParser("auto_load")
	parser.add_argument("-l", "--log_level", default="warning", help="Show at least this level of logs")
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
		config = validate_configuration(args.config, "meerkat/classification/config/auto_load_schema.json")
		aspirants = config.get("aspirants", {})
		args.region = config.get("region", args.region)
		args.prefix = config.get("prefix", args.prefix)
		args.bucket = config.get("bucket", args.bucket)
	logging.info("Aspirants are: {0}".format(aspirants))
	logging.info("Region: {0}".format(args.region))
	logging.info("Prefix: {0}".format(args.prefix))
	logging.info("Bucket: {0}".format(args.bucket))

	logging.warning("Starting main program")
	conn = connect_to_region(args.region)
	bucket = conn.get_bucket(args.bucket)
	my_results, target = {}, "results.tar.gz",
	find_s3_objects_recursively(conn, bucket, my_results, prefix=args.prefix, target=target)
	results = get_peer_models(my_results, prefix=args.prefix)
	get_best_models(bucket, args.prefix, results, target, args.prefix, aspirants)
	logging.warning("Finishing main program")

if __name__ == "__main__":
	#Execute the main program
	main_program()

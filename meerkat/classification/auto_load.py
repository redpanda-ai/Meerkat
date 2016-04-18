"""This module will update Meerkat's models from S3"""

import argparse
import boto
import logging
import re
import tarfile
import pandas as pd

from boto.s3.key import Key
from os import rename

def find_s3_objects_recursively(conn, bucket, my_results, prefix=None, target=None):
	"""Find all S3 target objects and their locations recursively"""
	folders = bucket.list(prefix=prefix, delimiter="/")
	for s3_object in folders:
		if s3_object.name != prefix:
			if s3_object.name[-len(target):] == target:
				my_results[prefix] = target
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
	#drop columns 0, 1, and -1 to make a square confusion matrix
	df = df.drop(df.columns[[0, 1, -1]], axis=1)
	#Reset columns names, which are off by one
	df.rename(columns=lambda x: int(x)-1, inplace=True)
	#get the diagonal of true positives as a vector
	rows, _ = df.shape

	#First order calculations (good calculations, mostly unused)
	true_positive = pd.DataFrame(df.iat[i, i] for i in range(rows))
	#col_sum = pd.DataFrame(df.sum(axis=1))
	#false_positive = pd.DataFrame(pd.DataFrame(df.sum(axis=0)).values - true_positive.values,
	#	columns=true_positive.columns)
	#false_negative = pd.DataFrame(pd.DataFrame(df.sum(axis=1)).values - true_positive.values,
	#	columns=true_positive.columns)
	#true_negative = pd.DataFrame(
	#	[df.drop(i, axis=1).drop(i, axis=0).sum().sum() for i in range(rows)])

	#Second order calculations (good calculations, mostly unused)
	accuracy = true_positive.sum() / df.sum().sum()
	#precision = true_positive / (true_positive + false_positive)
	#recall = true_positive / (true_positive + false_negative)
	#specificity = true_negative / (true_negative + false_positive)

	#Third order calculations (good calculation, unused)
	#f_measure = 2 * precision * recall / (precision + recall)
	#Return the model accuracy, which is all we care about, actually
	return accuracy.values[0]

def get_single_file_from_tarball(archive, filename_pattern):
	"""Untars and gunzips the stats file from the archive file"""
	if not tarfile.is_tarfile(archive):
		raise Exception("Invalid, not a tarfile.")
	my_pattern = re.compile(filename_pattern)
	with tarfile.open(name=archive, mode="r:gz") as tar:
		members = tar.getmembers()
		logging.debug("Members {0}".format(members))
		file_list = [member for member in members if my_pattern.search(member.name)]
		if len(file_list) != 1:
			logging.critical("Invalid, tarfile has several matching files.")
			raise Exception("Invalid, tarfile has several matching files.")
		else:
			my_file = file_list.pop()
			my_name = my_file.name
			tar.extract(my_file)
	return my_name

def get_best_models(bucket, prefix, results, target):
	"""Gets the best model for a particular model type."""
	for key in sorted(results.keys()):
		highest_score, winner = 0.0, None
		logging.info("Evaluating {0}".format(key))
		candidate_count = 1
		for timestamp in results[key]:
			k = Key(bucket)
			k.key = prefix + key + timestamp + target
			k.get_contents_to_filename(target)
			matrix = get_single_file_from_tarball(target, "Con_Matrix.csv")
			score = get_model_accuracy(matrix)
			if score > highest_score:
				highest_score = score
				winner = timestamp
				winner_count = candidate_count
				#Find the actual checkpoint file.
				leader = get_single_file_from_tarball(target, ".*ckpt")
				new_path = "meerkat/classification/models/" + key.replace("/", ".")[1:] + "ckpt"
				logging.debug("Moving label_map to: {0}".format(new_path))
				rename(leader, new_path)

			logging.info("\t{0:<14}{1:>2}: {2:16}, Score: {3:0.5f}".format(
				"Candidate", candidate_count, timestamp, score))
			candidate_count += 1
		set_label_map(bucket, prefix, key, winner, "input.tar.gz", "meerkat/classification/models/")
		logging.info("\t{0:<14}{1:>2}".format("Winner", winner_count))

def set_label_map(bucket, prefix, key, winner, tarball, output_path):
	"""Moves the appropriate label map from S3 to the local machine."""
	if bucket is not None:
		s3_key = Key(bucket)
		s3_key.key = prefix + key + winner + tarball
		s3_key.get_contents_to_filename(tarball)

	json_file = get_single_file_from_tarball(tarball, ".*json")
	new_path = output_path + key.replace("/", ".")[1:] + "json"
	logging.debug("Moving label_map to: {0}".format(new_path))
	rename(json_file, new_path)
	return new_path

def main_program():
	"""Execute the main program"""
	conn = boto.s3.connect_to_region('us-west-2')
	bucket = conn.get_bucket("s3yodlee")
	my_results, prefix, target = {}, "meerkat/cnn/data", "results.tar.gz"
	find_s3_objects_recursively(conn, bucket, my_results, prefix=prefix, target=target)
	results = get_peer_models(my_results, prefix=prefix)
	get_best_models(bucket, prefix, results, target)

if __name__ == "__main__":
	#Execute the main program
	parser = argparse.ArgumentParser("auto_load")
	parser.add_argument("-d", "--debug", help="Show 'debug'+ level logs", action="store_true")
	parser.add_argument("-v", "--info", help="Show 'info'+ level logs", action="store_true")
	args = parser.parse_args()
	log_format = "%(asctime)s %(levelname)s: %(message)s"
	if args.debug:
		logging.basicConfig(format=log_format, level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(format=log_format, level=logging.INFO)
	else:
		logging.basicConfig(format=log_format, level=logging.WARNING)
	logging.warning("Starting main program")
	main_program()
	logging.warning("Finishing main program")


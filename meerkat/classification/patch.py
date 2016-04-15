"""This module will update Meerkat's models from S3"""

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
		else:
			logging.warning("Not Found")
	return results

def show_dataframe(name, df):
	"""Display information about a dataframe."""
	logging.warning("{0}\n{1},\nShape {2},\nSize {3}".format(name, df, df.shape, df.size))

def get_model_accuracy(stats_file):
	"""Gets the true positive for a particular confusion matrix"""
	df = pd.read_csv(stats_file)
	#drop columns 0, 1, and -1 to make a square confusion matrix
	df = df.drop(df.columns[[0,1,-1]], axis=1)
	#Reset columns names, which are off by one
	df.rename(columns=lambda x: int(x)-1, inplace=True)
	#get the diagonal of true positives as a vector
	rows, _ = df.shape
	#First order calculations
	true_positive = pd.DataFrame(df.iat[i,i] for i in range(rows))
	col_sum = pd.DataFrame(df.sum(axis=1))
	false_positive = pd.DataFrame(pd.DataFrame(df.sum(axis=0)).values - true_positive.values,
		columns=true_positive.columns)
	false_negative = pd.DataFrame(pd.DataFrame(df.sum(axis=1)).values - true_positive.values,
		columns=true_positive.columns)
	true_negative = pd.DataFrame(
		[df.drop(i, axis=1).drop(i, axis=0).sum().sum() for i in range(rows)])
	#Second order calculations
	accuracy = true_positive.sum() / df.sum().sum()
	precision = true_positive / (true_positive + false_positive)
	recall = true_positive / (true_positive + false_negative)
	specificity = true_negative / (true_negative + false_positive)
	#Third order calculations
	f_measure = 2 * precision * recall / (precision + recall)
	#Return the model accuracy, which is all we care about, actually
	return accuracy.values[0]

def get_archived_file(archive, archived_file):
	"""Untars and gunzips the stats file from the archive file"""
	if not tarfile.is_tarfile(archive):
		logging.warning("Invalid tarfile")
		return None
	my_pattern = re.compile(archived_file)
	with tarfile.open(name=archive, mode="r:gz") as tar:
		members = tar.getmembers()
		stats = [ member for member in members if my_pattern.search(member.name) ]
		if len(stats) != 1:
			logging.critical("Invalid tarfile, must contain exactly 1 archived_file")
			logging.critical("Invalid Tarfile contains {0}".format(stats))
			raise Exception("Invalid tarfile.")
		else:
			my_file = stats.pop()
			my_name = my_file.name
			tar.extract(my_file)
	return my_name

def get_best_models(bucket, prefix, results, target):
	"""Gets the best model for a particular model type."""
	model_base = "meerkat/classification/models/"
	winners = {}
	for key in sorted(results.keys()):
		highest_score, winner = 0.0, None
		result_format = "\t{0:<14}: {1:16}, Score: {2:0.5f}"
		logging.warning("Evaluating {0}".format(key))
		for timestamp in results[key]:
			k = Key(bucket)
			k.key = prefix + key + timestamp + target
			k.get_contents_to_filename(target)
			matrix = get_archived_file(target, "Con_Matrix.csv")
			score = get_model_accuracy(matrix)
			if score > highest_score:
				highest_score = score
				winner = timestamp
				#Find the actual checkpoint file.
				leader = get_archived_file(target, ".*ckpt")
				rename(leader, model_base + key.replace("/",".") + "ckpt")
			logging.warning(result_format.format("Candidate", timestamp, score))
		winners[key] = winner
		logging.warning(result_format.format("Winner", winner, highest_score))
	return winners

def main_patch():
	"""Execute the main program"""
	conn = boto.s3.connect_to_region('us-west-2')
	bucket = conn.get_bucket("s3yodlee")
	#REVERT HERE my_results, prefix = {}, "meerkat/cnn/data"
	my_results, prefix = {}, "meerkat/cnn/data/"
	target = "results.tar.gz"
	find_s3_objects_recursively(conn, bucket, my_results, prefix=prefix, target=target)
	results = get_peer_models(my_results, prefix=prefix)
	winners = get_best_models(bucket, prefix, results, target)
	for winner in sorted(winners.keys()):
		logging.warning("Category: {0:<25}, Winner: {1}".format(winner, winners[winner]))

if __name__ == "__main__":
	#Execute the main program
	logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
	main_patch()


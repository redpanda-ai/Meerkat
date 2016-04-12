"""This module will update Meerkat's models from S3"""

import boto
import re
import tarfile
import pandas as pd

from boto.s3.key import Key

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
			print("Not Found")
	return results

def get_f_measure(stats_file):
	"""Evaluates a single stats_file and returns the f_measure"""
	df = pd.read_csv(stats_file)
	true_pos, false_neg = df["True_Positive"], df["False_Negative"]
	sums = df.sum(axis=0)
	#FIXME: I don't think this calcuation is meaningful, double-check it
	precision = sums["True_Positive"] / (sums["True_Positive"] + sums["False_Positive"])
	recall = sums["True_Positive"] / (sums["True_Positive"] + sums["False_Negative"])
	f_measure = 2 * precision * recall / (precision + recall)
	result_format = "{0:<12}: {1:0.5f}"
	#print("Sums {0}".format(sums))
	#print(result_format.format("Precision", precision))
	#print(result_format.format("Recall", recall))
	#print(result_format.format("F-measure", f_measure))
	return f_measure

def get_stats_file(target):
	"""Untars and gunzips the stats file from the target file"""
	stats_file = "CNN_stat.csv"
	if not tarfile.is_tarfile(target):
		print("Invalid tarfile")
		return None
	with tarfile.open(name=target, mode="r:gz") as tar:
		members = tar.getmembers()
		stats = [ member for member in members if member.name == stats_file ]
		if len(stats) != 1:
			print("Invalid tarfile, must contain exactly 1 stats_file")
			return None
		else:
			tar.extract(stats.pop())
	return stats_file

def get_best_models(bucket, prefix, results, target):
	"""Gets the best model for a particular model type."""
	winners = {}
	for key in sorted(results.keys()):
		highest_score, winner = 0.0, None
		result_format = "\t{0:<14}: {1:16}, Score: {2:0.5f}"
		print("Category {0}".format(key))
		for timestamp in results[key]:
			k = Key(bucket)
			k.key = prefix + key + timestamp + target
			k.get_contents_to_filename(target)
			score = get_f_measure(get_stats_file(target))
			if score > highest_score:
				highest_score = score
				winner = timestamp
			print(result_format.format("Candidate", timestamp, score))
		winners[key] = winner
		print(result_format.format("Winner", winner, highest_score))
	return winners

def main_patch():
	"""Execute the main program"""
	conn = boto.s3.connect_to_region('us-west-2')
	bucket = conn.get_bucket("s3yodlee")
	my_results, prefix = {}, "meerkat/cnn/data"
	target = "results.tar.gz"
	find_s3_objects_recursively(conn, bucket, my_results, prefix=prefix, target=target)
	results = get_peer_models(my_results, prefix=prefix)
	winners = get_best_models(bucket, prefix, results, target)
	for winner in sorted(winners.keys()):
		print("Category: {0:<25}, Winner: {1}".format(winner, winners[winner]))

if __name__ == "__main__":
	#Execute the main program
	main_patch()


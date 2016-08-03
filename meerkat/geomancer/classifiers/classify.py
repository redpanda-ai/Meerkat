#################### USAGE ##########################

#Run the module with -h to get help

#####################################################

import argparse
import csv
import logging
import re
import sys
import numpy as np
import pandas as pd


from .bloom_filter import BloomfilterClassifier
from .trie import TrieClassifier
from sklearn.metrics import classification_report

def parse_arguments():
	"""Parses arguments from the command-line"""
	parser = argparse.ArgumentParser()
	parser.add_argument("input_csv_file")
	parser.add_argument("classifier")
	return parser.parse_args()

def format_classification_report(report):
	"""Make things easier to read."""
	report_format = " {0:<20} {1:<10} {2:<10} {3:<10} {4:<10}"
	pattern = "^(.+,\S\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+).*$"
	digits_pattern = "^(\d+\S+)\s+(\d+\S+)\s+(\d+\S+)\s+(\d+\S+)\s+(\d+\S+).*$"
	summary_pattern = "^(avg / total)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+).*$"
	first_pattern = "^(\S+)\s+(\S+)\s+(\S+)\s+(\S+).*$"

	lines = ["\n"]
	my_regex = re.compile(pattern)
	my_digits = re.compile(digits_pattern)
	my_summary = re.compile(summary_pattern)
	my_first = re.compile(first_pattern)
	for line in report.split("\n"):
		line = line.strip()
		dig_match = my_digits.match(line)
		sum_match = my_summary.match(line)
		first_match = my_first.match(line)

		matches = my_regex.match(line)
		if matches:
			lines.append(report_format.format(matches.group(1), matches.group(2), matches.group(3), matches.group(4), matches.group(5)))
		elif dig_match:
			lines.append(report_format.format("", dig_match.group(1), dig_match.group(2), dig_match.group(3), dig_match.group(4)))
		elif sum_match:
			lines.append(report_format.format(sum_match.group(1), sum_match.group(2), sum_match.group(3), sum_match.group(4), sum_match.group(5)))
		elif first_match:
			lines.append(report_format.format("", first_match.group(1), first_match.group(2), first_match.group(3), first_match.group(4)))
	logging.info(str.join("\n", lines))

def main_process():
	"""This does all the work."""
	logging.basicConfig(level=logging.INFO)

	args = parse_arguments()
	input_csv_file = args.input_csv_file
	classifier = args.classifier

	if classifier == "bloomfilter":
		classifier = BloomfilterClassifier()
	elif classifier == "trie":
		classifier = TrieClassifier()
	else:
		logging.error("Wrong classifier name")
		sys.exit()

	df = pd.read_csv(input_csv_file, error_bad_lines=False,
		encoding='utf-8', quoting=csv.QUOTE_NONE, na_filter=False, sep='\t')

	df['city_state'] = df[['city', 'state']].apply(lambda x: ','.join(x), axis=1)

	# fit
	msk = np.random.rand(len(df)) < 0.80
	train = df[msk]
	test = df[~msk]

	X_train = []
	for index, row in train.iterrows():
		X_train.append([row['question']])
	y_train = list(train['city_state'])

	classifier.fit(X_train, y_train)

	# predict
	X = df['question']
	y_predict = classifier.predict(X)

	y_true = []
	for item in list(df['city_state']):
		y_true.append(item if str(item) != ',' else '')

	descriptions = [item for item in list(df['question'])]

	true_positives = 0
	logging.info("{:<20}".format("Predicted") + "{:<20}".format("Labeled") + "{:<80}".format("Description"))
	for i in range(len(y_predict)):
		logging.info("{:<20}".format(y_predict[i]) + "{:<20}".format(y_true[i]) + "{:<80}".format(descriptions[i]))
		if y_predict[i] == y_true[i]:
			true_positives += 1


	# generate classification report
	report = classification_report(y_true, y_predict)
	format_classification_report(report)
		#columns = line.split("\t")
		#columns = [x.strip() for x in columns]
		#if len(columns) >=4:
		#	logging.info(report_format.format(columns[0],columns[1],columns[2],columns[3]))
		#else:
		#	logging.info(columns)
		#logging.info("! {0}".format(line))
	#logging.info(report)

	# calculate accuracy
	accuracy = true_positives / len(y_predict)
	logging.info("Accuray is: {:.2%}".format(accuracy))

if __name__ == "__main__":
	main_process()

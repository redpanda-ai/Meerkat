#################### USAGE ##########################

# python3 -m meerkat.fat_head.classification.classify aligned.csv bloomfilter
# python3 -m meerkat.fat_head.classification.classify aligned.csv trie

#####################################################

import csv
import logging
import pandas as pd
import numpy as np
import argparse
import sys

from meerkat.fat_head.classification.bloomfilter_classifier import BloomfilterClassifier
from meerkat.fat_head.classification.trie_classifier import TrieClassifier
from sklearn.metrics import classification_report

def parse_arguments():
	"""Parses arguments from the command-line"""
	parser = argparse.ArgumentParser()
	parser.add_argument("input_csv_file")
	parser.add_argument("classifier")
	return parser.parse_args()

def main_process():
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
	msk = np.random.rand(len(df)) < 0.80
	train = df[msk]
	test = df[~msk]

	X_train = []
	for index, row in train.iterrows():
		X_train.append([row['question']])
	y_train = list(train['city_state'])

	classifier.fit(X_train, y_train)

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

	report = classification_report(y_true, y_predict)
	logging.info(report)

	accuracy = true_positives / len(y_predict)
	logging.info("Accuray is: {:.2%}".format(accuracy))

if __name__ == "__main__":
	main_process()

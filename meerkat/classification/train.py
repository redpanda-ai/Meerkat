#!/usr/local/bin/python3.3
# pylint: disable=bad-continuation
# pylint: disable=no-value-for-parameter
# pylint: disable=unexpected-keyword-arg

"""This module generates and trains a binary classification model for
transactions using SciKit Learn. Specifically it uses a Stochastic
Gradient Descent classifier that is optimized using Grid Search

Created on Feb 25, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 -m meerkat.classification.train [labeled_file] [label_key]
# python3.3 -m meerkat.classification.train data/input/matt_8000_card.txt IS_PHYSICAL_TRANSACTION

#####################################################

import csv
import sys
import logging
import os
from random import random

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

def split_data(labeled_trans="data/misc/matt_8000_card.csv"):
	"""Divides the training set into parts for testing and training."""

	if not os.path.isfile(labeled_trans):
		logging.error("Please provide a set of labeled transactions to build the classifier on")

	trans = []
	labels = []

	# Load Data
	trans, labels = load_data(trans, labels, labeled_trans)

	# Append More
	#trans, labels = load_data(trans, labels, "data/misc/verifiedLabeledTrans.csv")

	print("NUMBER OF TRANSACTIONS: ", len(trans))

	# Split into training and testing sets
	if len(trans) < 100:
		logging.error("Not enough labeled data to create a model from")

	split_results = train_test_split(trans, labels, test_size=0.2)
	trans_train, trans_test, labels_train, labels_test = split_results

	return trans_train, trans_test, labels_train, labels_test

def load_data(trans, labels, file_name, test_size=1):
	"""Loads human labeled data from a file."""

	human_labeled_file = open(file_name, encoding='utf-8', errors="replace")
	human_labeled = list(csv.DictReader(human_labeled_file, delimiter='|'))
	human_labeled_file.close()

	for label in human_labeled:
		if label[sys.argv[2]] != "" and random() < test_size:
			trans.append(label["DESCRIPTION_UNMASKED"])
			labels.append(label[sys.argv[2]])

	return trans, labels

def build_model(trans_train, trans_test, labels_train, labels_test):
	"""Creates a classifier using the training set and then scores the
	result."""

	pipeline = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier())
	])

	parameters = {
		'vect__max_df': (0.05, 0.10, 0.25, 0.5),
		'vect__max_features': (1000, 2000, 3000, 4000, 5000, 6000),
 		'vect__ngram_range': ((1, 1), (1, 2)), # unigrams or bigrams
		'tfidf__use_idf': (True, False),
		'tfidf__norm': ('l1', 'l2'),
		'clf__alpha': (0.00001, 0.0000055, 0.000001),
		'clf__penalty': ('l2', 'elasticnet'),
		'clf__n_iter': (10, 50, 80)
	}

	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=4)
	grid_search.fit(trans_train, labels_train)
	score = grid_search.score(trans_test, labels_test)

	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")

	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	print("Actual Score: " + str(score))

	# Save Model
	model_name = 'meerkat/classification/models/new_model' + sys.argv[2] + '.pkl'
	joblib.dump(grid_search, model_name, compress=3)

def test_model(file_to_test, model):
	"""Tests our classifier."""
	trans, labels = load_data([], [], file_to_test)
	score = model.score(trans, labels)

	print(file_to_test, " Score: ", score)

def run_from_command_line(cla):
	"""Runs the module when invoked from the command line."""

	if len(cla) == 3 and os.path.isfile(cla[1]):
		trans_train, trans_test, labels_train, labels_test = split_data(labeled_trans=cla[1])
	else:
		trans_train, trans_test, labels_train, labels_test = split_data()

	build_model(trans_train, trans_test, labels_train, labels_test)

if __name__ == "__main__":
	run_from_command_line(sys.argv)

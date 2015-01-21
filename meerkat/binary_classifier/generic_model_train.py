#!/usr/local/bin/python3.3

"""This module generates and trains classificaton models for
transactions using SciKit Learn. Specifically it uses a Stochastic 
Gradient Descent classifier that is optimized using Grid Search

Created on Feb 25, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Experts only! Do not touch!
# python3.3 generic_model_train.py [file_name] [label_column_name] [document_column_name]

#####################################################

import csv
import sys
import logging
import os
from random import random

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from sklearn.base import TransformerMixin

class ColumnExtractor(TransformerMixin):

    def __init__(self, columns=[]):
        self.columns = columns

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def transform(self, X, **transform_params):
        return X[self.columns]

    def fit(self, X, y=None, **fit_params):
        return self

def split_data(labeled_transactions):
	"""Divides the training set into parts for testing and training."""
	if not os.path.isfile(labeled_transactions):
		logging.error("Please provide a set of labeled transactions to"\
			+ "build the classifier on")

	transactions = []
	labels = []

	# Load Data
	transactions, labels = load_data(transactions, labels, labeled_transactions)

	# Append More
	#transactions, labels = load_data(transactions,
	#labels, "data/misc/verifiedLabeledTrans.csv")

	print("NUMBER OF TRANSACTIONS: ", len(transactions))

	# Split into training and testing sets
	if len(transactions) < 100:
		logging.error("Not enough labeled data to create a model from")

	trans_train, trans_test, labels_train,\
		labels_test = train_test_split(transactions, labels, test_size=0.2)

	return trans_train, trans_test, labels_train, labels_test

def load_data(transactions, labels, file_name):
	"""Loads human labeled data from a file."""

	human_labeled_file = open(file_name, encoding='utf-8', errors="replace")
	human_labeled = list(csv.DictReader(human_labeled_file, delimiter='|'))
	human_labeled_file.close()
	label_col_name = sys.argv[2]
	doc_col_name = sys.argv[3]

	for i in range(len(human_labeled)):
		transactions.append(human_labeled[i][doc_col_name])
		labels.append(human_labeled[i][label_col_name].upper())

	return transactions, labels

def build_model(trans_train, trans_test, labels_train, labels_test):
	"""Creates a classifier using the training set and then scores the
	result."""

	pipeline = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier())
	])

	parameters = {
		'vect__max_df': (0.25, 0.35),
		'vect__max_features': (None, 9000),
		'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
		'tfidf__use_idf': (True, False),
		'tfidf__norm': ('l1', 'l2'),
		'clf__alpha': (0.0000055, 0.000008),
		'clf__penalty': ('l2', 'elasticnet'),
		'clf__n_iter': (50, 80)
	}

	grid_search = GridSearchCV(pipeline, parameters, n_jobs=8, verbose=3, cv=3)
	grid_search.fit(trans_train, labels_train)
	score = grid_search.score(trans_test, labels_test)

	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	print("Actual Score: " + str(score))

	# Test Model
	#test_model("100000_non_physical_bank.txt", grid_search)

	# Save Model
	joblib.dump(grid_search, 'meerkat_card_model_' + str(score) + '.pkl', compress=3)

def test_model(file_to_test, model):
	"""Tests our classifier."""
	transactions, labels = load_data([], [], file_to_test)
	score = model.score(transactions, labels)

	print(file_to_test, " Score: ", score)

def run_from_command_line(command_line_arguments):
	"""Runs the module when invoked from the command line."""
	if len(command_line_arguments) == 4\
	and os.path.isfile(command_line_arguments[1]):
		trans_train, trans_test, labels_train, labels_test =\
			split_data(labeled_transactions=command_line_arguments[1])
	else:
		trans_train, trans_test, labels_train, labels_test = split_data()
	build_model(trans_train, trans_test, labels_train, labels_test)

if __name__ == "__main__":
	run_from_command_line(sys.argv)

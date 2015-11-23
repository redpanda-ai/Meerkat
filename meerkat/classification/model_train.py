#!/usr/local/bin/python3.3

"""This module generates and trains classificaton models for
transactions using SciKit Learn. Specifically it uses a Stochastic
Gradient Descent classifier that is optimized using Grid Search

Created on Feb 25, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Experts only! Do not touch!
# python3 -m meerkat.classification.model_train [file_name] [label_column_name]

#####################################################

import csv
import sys
import logging
import os
#Import below is unused
#import math
from random import random

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnExtractor(BaseEstimator, TransformerMixin):
	#pylint:disable=invalid-name,unused-argument
	"""This extracts columns.
		X is the trainging set.
		y is the target values.
		**fit_params is an optional parameter
		that returns a transformed version of training set.
	"""
	def __init__(self, columns=[]):
		self.columns = columns

	def transform(self, X, **transform_params):
		return X[:, self.columns]

	def fit(self, X, y=None, **fit_params):
		return self

class DateTransformer(BaseEstimator, TransformerMixin):
	#pylint:disable=invalid-name,unused-argument
	"""This transforms dates.
		X is the trainging set.
		y is the target values.
		**fit_params is an optional parameter
		that returns a transformed version of training set.
	"""

	def __init__(self, columns=[]):
		pass

	def transform(self, X, **transform_params):
		dict_list = [dict(zip(("TYPE"), x[-2:])) for x in X]
		return dict_list

	def fit(self, X, y=None, **fit_params):
		return self

def split_data(filename):
	"""Divides the training set into parts for testing and training."""

	if not os.path.isfile(filename):
		logging.error("Please provide a set of labeled transactions to"\
			+ "build the classifier on")

	# Load Data
	transactions, labels = load_data_list([], [], filename)

	# Split into training and testing sets
	if len(transactions) < 100:
		logging.error("Not enough labeled data to create a model from")

	return transactions, labels

def load_data_df(file_name):
	"""Loads human labeled data from a file."""

	df = pd.read_csv(file_name, na_filter=False, parse_dates=["TRANSACTION_DATE"],\
	 quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
	label_col_name = sys.argv[2]

	transactions = df[['DESCRIPTION_UNMASKED', 'TRANSACTION_DATE', 'TYPE']]
	labels = df[label_col_name]

	return transactions, list(labels)

def load_data_list(transactions, labels, file_name, test_size=1):
	"""Loads human labeled data from a file."""

	label_col_name = sys.argv[2]
	human_labeled_file = open(file_name, encoding='utf-8', errors="replace")
	human_labeled = list(csv.DictReader(human_labeled_file, delimiter='|'))
	human_labeled_file.close()

	for i in range(len(human_labeled)):
		if human_labeled[i][label_col_name] != "" and random() < test_size:
			transactions.append(human_labeled[i]["DESCRIPTION_UNMASKED"])
			labels.append(human_labeled[i][label_col_name])

	return transactions, labels

def extract_features(transactions):
	"""Extract features for dict vectorizer"""

	l = 'TRANSACTION_DATE'
	f = lambda x: str(x[l].weekday()) if not isinstance(x[l], float) else ""
	g = lambda x: str(x[l].day) if not isinstance(x[l], float) else ""
	transactions['WEEKDAY'] = transactions.apply(f, axis=1)
	transactions['DAYOFMONTH'] = transactions.apply(g, axis=1)

	return transactions

def build_model(transactions, labels):
	"""Creates a classifier using the training set and then scores the
	result."""

	simple_parameters = {
		'vect__max_df': (0.05, 0.10, 0.25, 0.5),
		'vect__max_features': (1000, 2000, 3000, 4000, 5000, 6000),
 		'vect__ngram_range': ((1, 1), (1, 2)), # unigrams or bigrams
		'tfidf__use_idf': (True, False),
		'tfidf__norm': ('l1', 'l2'),
		'clf__alpha': (0.00001, 0.0000055, 0.000001),
		'clf__penalty': ('l2', 'elasticnet'),
		'clf__n_iter': (10, 50, 80)
	}

	simple_pipeline = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier())
	])

	"""
	complex_pipeline = Pipeline([
		('features', FeatureUnion([
			('bag_of_words', Pipeline([
				('extract', ColumnExtractor(0)),
				('vect', CountVectorizer()),
				('tfidf', TfidfTransformer())
			])),
			('day_of_week', Pipeline([
				('extract', DateTransformer()),
				('dict', DictVectorizer())
			]))
		])),
		('clf', SGDClassifier())
	])

	transactions = extract_features(transactions)

	complex_parameters = {
		'features__bag_of_words__vect__max_df': (0.25, 0.35),
		'features__bag_of_words__vect__max_features': (8500, None),
		'features__bag_of_words__vect__ngram_range': ((1, 1), (1, 2)), 
		# unigrams or bigrams
		'features__bag_of_words__tfidf__use_idf': (True, False),
		'features__bag_of_words__tfidf__norm': ('l1', 'l2'),
		'clf__alpha': (0.0000055, 0.000008),
		'clf__penalty': ('l2', 'elasticnet'),
		'clf__n_iter': (50, 80)
	}
	"""

	grid_search = GridSearchCV(simple_pipeline, simple_parameters, n_jobs=8,\
	 verbose=3, cv=3)
	grid_search.fit(transactions, labels)

	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(simple_parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	# Log Top Score
	score = grid_search.score(transactions, labels)
	print("Actual Score: " + str(score))

	# Save Model
	joblib.dump(grid_search, 'meerkat/classification/models/STO_bank_model_' +\
	 str(score) + '.pkl', compress=3)

	# Test Model
	#test_model("100000_non_physical_bank.txt", grid_search)

def test_model(file_to_test, model):
	"""Tests our classifier."""
	transactions, labels = load_data([], [], file_to_test)
	score = model.score(transactions, labels)

	print(file_to_test, " Score: ", score)

def run_from_command_line(command_line_arguments):
	"""Runs the module when invoked from the command line."""
	if len(command_line_arguments) == 3:
		data = split_data(command_line_arguments[1])
		build_model(*data)
	else:
		print("Incorrect number of arguments")

if __name__ == "__main__":
	run_from_command_line(sys.argv)

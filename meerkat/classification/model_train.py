#!/usr/local/bin/python3.3

"""This module generates and trains classificaton models for
transactions using SciKit Learn. Specifically it uses a Stochastic 
Gradient Descent classifier that is optimized using Grid Search

Created on Feb 25, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Experts only! Do not touch!
# python3.3 -m meerkat.classification.model_train [file_name] [label_column_name]

#####################################################

import csv
import sys
import logging
import os
import math
from random import random

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnExtractor(BaseEstimator, TransformerMixin):

	def __init__(self, columns=[]):
		self.columns = columns

	def transform(self, X, **transform_params):
		return X[:, self.columns]

	def fit(self, X, y=None, **fit_params):
		return self

class DateTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, columns=[]):
		pass

	def transform(self, X, **transform_params):
		dict_list = [dict(zip(("TYPE"), x[-2:])) for x in X]
		return dict_list

	def fit(self, X, y=None, **fit_params):
		return self

def split_data(labeled_transactions):
	"""Divides the training set into parts for testing and training."""
	
	if not os.path.isfile(labeled_transactions):
		logging.error("Please provide a set of labeled transactions to"\
			+ "build the classifier on")

	# Load Data
	transactions, labels = load_data(labeled_transactions)

	# Split into training and testing sets
	if len(transactions.index) < 100:
		logging.error("Not enough labeled data to create a model from")

	return transactions, labels

def load_data(file_name):
	"""Loads human labeled data from a file."""

	df = pd.read_csv(file_name, na_filter=False, parse_dates=["TRANSACTION_DATE"], quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
	label_col_name = sys.argv[2]

	transactions = df[['DESCRIPTION_UNMASKED', 'TRANSACTION_DATE', 'TYPE']]
	labels = df[label_col_name]

	return transactions, list(labels)

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

	transactions = extract_features(transactions)

	parameters = {
		'features__bag_of_words__vect__max_df': (0.25, 0.35),
		'features__bag_of_words__vect__max_features': (8500, None),
		'features__bag_of_words__vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
		'features__bag_of_words__tfidf__use_idf': (True, False),
		'features__bag_of_words__tfidf__norm': ('l1', 'l2'),
		'clf__alpha': (0.0000055, 0.000008),
		'clf__penalty': ('l2', 'elasticnet'),
		'clf__n_iter': (50, 80)
	}

	pipeline = Pipeline([
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

	grid_search = GridSearchCV(pipeline, parameters, n_jobs=8, verbose=3, cv=3)
	grid_search.fit(transactions, labels)

	# TEMP
	joblib.dump(grid_search, 'category_model_.pkl', compress=3)

	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	# Log Top Score
	score = grid_search.score(transactions, labels)
	print("Actual Score: " + str(score))

	# Save Model
	joblib.dump(grid_search, 'category_model_' + str(score) + '.pkl', compress=3)

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

#!/usr/local/bin/python3
# pylint: disable=C0301

"""This module performs hyperparameter optimization. This involves
tuning the keys located under Longtail/config/keys. Those key value
pairs map to hyperparameters used through out the Longtail Classifier.
This module utilizes a common method known as grid search. In particular
we are using randomized optimization as it works better where it
is resource intensive to exaustively perform a standard grid_search"""

from longtail.description_producer import initialize, get_desc_queue, tokenize, load_parameter_key

import sys, pprint
from random import randint, uniform
from sklearn.grid_search import RandomizedSearchCV

def recurse(iter=5):

	"""Run randomized_optimization until the distributions shrink"""	

def randomized_optimization(iter=10, min_precision=99, min_recall=35):

	"""Generates randomized parameter keys by
	providing a range to sample from. 
	Runs the classifier a fixed number of times and
	provides the top score found"""

	# Init
	params = initialize()
	results = []

	# Run a provided number of times
	for i in range(iter):

		print("\n", "ITERATION NUMBER: " + str(i))

		randomized_hyperparameters = {
			"es_result_size" : randint(45, 45),
			"z_score_threshold" : uniform(0.409, 1.109),
			"business_name_boost" : uniform(0.546, 1.146),
			"city_name_boost" : uniform(2.166, 2.766),
			"street_boost" : uniform(0.518, 1.118),
		}

		# Get Randomized Hyperparameters
		for key, value in randomized_hyperparameters.items():
			if type(value) == int:
				randomized_hyperparameters[key] = str(value)
			elif type(value) == float:
				randomized_hyperparameters[key] = str(round(value, 3))
	
		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		desc_queue, non_physical = get_desc_queue(params)
		accuracy = tokenize(params, desc_queue, randomized_hyperparameters, non_physical)

		print("\n", randomized_hyperparameters)

		# Save Good Hyperparameters
		if accuracy['precision'] >= min_precision and accuracy['total_recall'] >= min_recall:
			accuracy['hyperparameters'] = randomized_hyperparameters
			results.append(accuracy)
			pprint.pprint(accuracy, record)

	# We need at least 1 result
	if len(results) < 1:
		randomized_optimization(iter=1)
		return

	# Get Top Score
	top_score = {"total_recall":0}
	hyperparameters_used = [result['hyperparameters'] for result in results]

	for score in results:
		if score["total_recall"] > top_score["total_recall"]:
			top_score = score

	print("Precision = " + str(top_score['precision']) + "%")
	print("Best Recall = " + str(top_score['total_recall']) + "%")
	print("HYPERPARAMETERS:")
	print(top_score["hyperparameters"])
	print("ALL RESULTS:")
	
	print(hyperparameters_used)

	return hyperparameters_used

if __name__ == "__main__":

	record = open("initialKeys.txt", "a")
	randomized_optimization(iter=2000)

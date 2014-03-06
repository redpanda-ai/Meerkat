#!/usr/local/bin/python3
# pylint: disable=all

"""This module performs hyperparameter optimization. This involves
tuning the keys located under Longtail/config/keys. Those key value
pairs map to hyperparameters used through out the Longtail Classifier.
This module utilizes a common method known as grid search. In particular
we are using randomized optimization as it works better where it
is resource intensive to exaustively perform a standard grid_search"""

from longtail.description_producer import initialize, get_desc_queue, tokenize, load_hyperparameters

import sys, pprint
from random import randint, uniform
from sklearn.grid_search import RandomizedSearchCV


def get_initial_values(hyperparameters, params, known, iter=100):
	"""Do a simple search to find starter values"""

	for i in range(iter):
		randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=1)
		print("\n", "ITERATION NUMBER: " + str(i))
		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params)
		top_score = {"precision" : 0}
		if accuracy["precision"] >= top_score["precision"]:
			top_score = accuracy
			print("\n", "SCORE: " + str(accuracy["precision"]))

		print("\n", randomized_hyperparameters,"\n")

	print("TOP SCORE:" + str(top_score["precision"]))
	return top_score

def randomize(hyperparameters, known={}, learning_rate=0.3):
	"""Finds new values within a given range 
	based on a provided learning rater"""

	randomized = {}

	for key, value in hyperparameters.items():
		lower = float(value) - learning_rate
		lower = lower if lower >= 0 else 0
		upper = float(value) + learning_rate
		upper = upper if upper <=3 else 3
		new_value = uniform(lower, upper)
		randomized[key] = str(round(new_value, 1))

	return dict(list(randomized.items()) + list(known.items()))


def run_classifier(hyperparameters, params):
	""" Runs the classifier with a given set of hyperparameters"""

	desc_queue, non_physical = get_desc_queue(params)
	accuracy = tokenize(params, desc_queue, hyperparameters, non_physical)

	return accuracy

def randomized_optimization(hyperparameters, known, params, iter=10, min_precision=99, min_recall=35):

	"""Generates randomized parameter keys by
	providing a range to sample from. 
	Runs the classifier a fixed number of times and
	provides the top score found"""

	get_initial_values(hyperparameters, params, known)
	sys.exit()

	# Init
	results = []

	# Run a provided number of times
	for i in range(iter):

		print("\n", "ITERATION NUMBER: " + str(i))

		randomized_hyperparameters = randomize(hyperparameters, known)

		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params)

		# Save Good Hyperparameters
		if accuracy['precision'] >= min_precision and accuracy['total_recall'] >= min_recall:
			accuracy['hyperparameters'] = randomized_hyperparameters
			results.append(accuracy)
			pprint.pprint(accuracy, record)

	# We need at least 1 result
	if len(results) < 1:
		randomized_optimization(hyperparameters, known, params, iter=1)
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

	params = initialize()
	known = {"es_result_size" : "45"}
	record = open("initialKeys.txt", "a")

	hyperparameters = {
		"factual_id" : "1",      
	    "name" : "1",             
	    "address" : "1",          
	    "address_extended" : "1", 
	    "po_box" : "1",           
	    "locality" : "1",         
	    "region" : "1",           
	    "post_town" : "1",        
	    "admin_region" : "1",     
	    "postcode" : "1",         
	    "country" : "1",          
	    "tel" : "1",              
	    "fax" : "1",              
	    "neighborhood" : "1",     
	    "email" : "1",            
	    "category_ids" : "1",     
	    "category_labels" : "1",  
	    "status" : "1",          
	    "chain_name" : "1",       
	    "chain_id" : "1",
	    "pin.location" : "1",   
	    "composite.address" : "1",
		"z_score_threshold" : "1",
	}

	randomized_optimization(hyperparameters, known, params, iter=3)
	record.close()

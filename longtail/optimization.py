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

	top_score = {"precision" : 0, "total_recall" : 0}

	for i in range(iter):
		randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=0.5)
		print("\n", "ITERATION NUMBER: " + str(i))
		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params)
		if accuracy["precision"] >= top_score["precision"] and accuracy["precision"] < 100:
			top_score = accuracy
			top_score['hyperparameters'] = randomized_hyperparameters
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
		randomized[key] = str(round(new_value, 3))

	return dict(list(randomized.items()) + list(known.items()))


def run_classifier(hyperparameters, params):
	""" Runs the classifier with a given set of hyperparameters"""

	boost_vectors = {}
	boost_labels = ["standard_fields"]
	other = {}

	for key, value in hyperparameters.items():
		if key == "es_result_size" or key == "z_score_threshold":
			other[key] = value
		else:
			boost_vectors[key] = [value]

	# Override Params
	params["elasticsearch"]["boost_labels"] = boost_labels
	params["elasticsearch"]["boost_vectors"] = boost_vectors

	# Run Classifier with new Hyperparameters
	desc_queue, non_physical = get_desc_queue(params)
	accuracy = tokenize(params, desc_queue, other, non_physical)

	return accuracy

def run_iteration(top_score, params, known, iter=100, convergence=1):

	hyperparameters = top_score['hyperparameters']
	new_top_score = top_score
	learning_rate = 0.3 * convergence

	for i in range(iter):
		randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=learning_rate)
		print("\n", "ITERATION NUMBER: " + str(i))
		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params)
		if accuracy["total_recall"] >= new_top_score["total_recall"] and accuracy["precision"] >= new_top_score["precision"] and accuracy["precision"] < 100:
			new_top_score = accuracy
			new_top_score['hyperparameters'] = randomized_hyperparameters
			print("\n", "SCORE: " + str(accuracy["precision"]))

	# We need at least 1 result
	#if len(results) < 1:
	#	run_iteration(hyperparameters, known, params, iter=1)
	#	return

	return new_top_score

def gradient_ascent(initial_values, params, known, iter=10):

	top_score = initial_values
	pprint.pprint(top_score, record)

	for i in range(iter):
		top_score = run_iteration(top_score, params, known, iter=iter, convergence=0.9)
		pprint.pprint(top_score, record)

	return top_score

def randomized_optimization(hyperparameters, known, params):

	"""Generates randomized parameter keys by
	providing a range to sample from. 
	Runs the classifier a fixed number of times and
	provides the top score found"""

	# Init
	initial_values = get_initial_values(hyperparameters, params, known, iter=15)

	# Run Gradient Ascent 
	top_score = gradient_ascent(initial_values, params, known, iter=15)

	print("Precision = " + str(top_score['precision']) + "%")
	print("Best Recall = " + str(top_score['total_recall']) + "%")
	print("HYPERPARAMETERS:")
	print(top_score["hyperparameters"])
	print("ALL RESULTS:")
	
	return top_score

if __name__ == "__main__":

	record = open("initialKeys.txt", "a")
	params = initialize()
	known = {"es_result_size" : "45"}

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

	found = {
		"chain_name": "1.7",
		"tel": "0.0",
		"name": "1.7",
		"locality": "1.5",
		"country": "0.1",
		"category_labels": "1.6",
		"status": "2.0",
		"region": "0.4",
		"address_extended": "0.4",
		"po_box": "1.3",
		"z_score_threshold": "1.1",
		"neighborhood": "1.0",
		"address": "0.8",
		"fax": "0.1",
		"admin_region": "0.9",
		"postcode": "0.8",
		"post_town": "0.6"
	} 

	randomized_optimization(hyperparameters, known, params)
	record.close()

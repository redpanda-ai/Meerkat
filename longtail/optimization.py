#!/usr/local/bin/python3
# pylint: disable=all

"""This module performs hyperparameter optimization. This involves
tuning the keys located under Longtail/config/keys. Those key value
pairs map to hyperparameters used through out the Longtail Classifier.
This module utilizes a common method known as grid search. In particular
we are using randomized optimization as it works better where it
is resource intensive to exaustively perform a standard grid_search"""

from longtail.description_producer import initialize, get_desc_queue, tokenize, load_hyperparameters

import sys, pprint, datetime
from random import randint, uniform
from sklearn.grid_search import RandomizedSearchCV


def get_initial_values(hyperparameters, params, known, iter=1):
	"""Do a simple search to find starter values"""

	top_score = {"precision" : 0, "total_recall_non_physical" : 0}
	learning_rate = settings["initial_learning_rate"]

	for i in range(iter):
		randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=learning_rate)
		print("\n", "ITERATION NUMBER: " + str(i))
		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params)
		higher_precision = accuracy["precision"] >= top_score["precision"]
		not_too_high = accuracy["precision"] <= settings["max_precision"]

		if higher_precision and not_too_high:
			top_score = accuracy
			top_score['hyperparameters'] = randomized_hyperparameters
			print("\n", "SCORE: " + str(accuracy["precision"]))

		print("\n", randomized_hyperparameters,"\n")

	if accuracy["precision"] < 90:
		get_initial_values(top_score["hyperparameters"], params, known, iter=1)

	print("TOP SCORE:" + str(top_score["precision"]))
	return top_score

def randomize(hyperparameters, known={}, learning_rate=0.3):
	"""Finds new values within a given range 
	based on a provided learning rate"""

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

def run_iteration(top_score, params, known, learning_rate):

	hyperparameters = top_score['hyperparameters']
	new_top_score = top_score
	learning_rate = learning_rate * settings["convergence"]
	iterations = settings["iteration_search_space"]

	for i in range(iterations):
		randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=learning_rate)
		print("\n", "ITERATION NUMBER: " + str(i))
		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params)
		same_or_higher_recall = accuracy["total_recall_non_physical"] >= new_top_score["total_recall_non_physical"]
		same_or_higher_precision = accuracy["precision"] >= new_top_score["precision"]
		not_too_high_precision = accuracy["precision"] <= settings["max_precision"]

		if same_or_higher_recall and same_or_higher_precision and not_too_high_precision:
			new_top_score = accuracy
			new_top_score['hyperparameters'] = randomized_hyperparameters
			print("\n", "SCORE: " + str(accuracy["precision"]))

	return new_top_score, learning_rate

def gradient_ascent(initial_values, params, known):

	top_score = initial_values
	learning_rate = settings["iteration_learning_rate"]
	save_top_score(initial_values)
	iterations = settings["gradient_ascent_iterations"]

	for i in range(iterations):
		print("LEARNING RATE: " + str(learning_rate))
		top_score, learning_rate = run_iteration(top_score, params, known, learning_rate)
		
		# Save Iterations Top Hyperparameters
		save_top_score(top_score)

	return top_score

def save_top_score(top_score):

	record = open("top_scores.txt", "a")
	pprint.pprint("Precision = " + str(top_score['precision']) + "%", record)
	pprint.pprint("Best Recall = " + str(top_score['total_recall_non_physical']) + "%", record)
	pprint.pprint(top_score["hyperparameters"], record)
	record.close()

def randomized_optimization(hyperparameters, known, params):

	"""Generates randomized parameter keys by
	providing a range to sample from. 
	Runs the classifier a fixed number of times and
	provides the top score found"""

	# Init
	iterations = settings["initial_search_space"]
	initial_values = get_initial_values(hyperparameters, params, known, iter=iterations)

	# Run Gradient Ascent 
	top_score = gradient_ascent(initial_values, params, known)

	print("Precision = " + str(top_score['precision']) + "%")
	print("Best Recall = " + str(top_score['total_recall_non_physical']) + "%")
	print("HYPERPARAMETERS:")
	pprint.pprint(top_score["hyperparameters"])
	print("ALL RESULTS:")
	
	# Save Final Parameters
	file_name = str(top_score['precision']) + "Precision" + str(top_score['total_recall_non_physical']) + "Recall.json" 
	new_parameters = open(file_name, 'w')
	pprint.pprint(top_score, new_parameters)

	return top_score

if __name__ == "__main__":

	# Clear Contents from Previous Runs
	open('top_scores.txt', 'w').close()

	start_time = datetime.datetime.now()
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
		"z_score_threshold" : "1"
	}

	settings = {
		"initial_search_space": 50,
		"initial_learning_rate": 1,
		"iteration_search_space": 25,
		"iteration_learning_rate": 0.3,
		"gradient_ascent_iterations": 15,
		"convergence": 0.9,
		"max_precision": 99
	}

	randomized_optimization(hyperparameters, known, params)
	time_delta = datetime.datetime.now() - start_time
	print("TOTAL TIME TAKEN FOR OPTIMIZATION: ", time_delta)
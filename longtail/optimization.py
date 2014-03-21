#!/usr/local/bin/python3
# pylint: disable=all

"""This module performs hyperparameter optimization. This involves
tuning the keys located under Longtail/config/keys. Those key value
pairs map to hyperparameters used through out the Longtail Classifier.
This module utilizes a common method known as grid search. In particular
we are using randomized optimization as it works better where it
is resource intensive to exaustively perform a standard grid_search"""

from longtail.description_producer import initialize, tokenize, load_hyperparameters
from longtail.binary_classifier.load import predict_if_physical_transaction
from longtail.accuracy import print_results

import sys, datetime, os, queue, csv
from pprint import pprint
from random import randint, uniform, random, shuffle
from numpy import array, array_split

def get_initial_values(hyperparameters, params, known, dataset):
	"""Do a simple search to find starter values"""

	top_score = {"precision" : 0, "total_recall_non_physical" : 0}
	iterations = settings["initial_search_space"]
	learning_rate = settings["initial_learning_rate"]

	for i in range(iterations):
		randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=learning_rate)
		print("\n", "ITERATION NUMBER: " + str(i))
		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params, dataset)
		higher_precision = accuracy["precision"] >= top_score["precision"]
		not_too_high = accuracy["precision"] <= settings["max_precision"]

		if higher_precision and not_too_high:
			top_score = accuracy
			top_score['hyperparameters'] = randomized_hyperparameters
			print("\n", "SCORE: " + str(accuracy["precision"]))

		print("\n", randomized_hyperparameters,"\n")

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


def run_classifier(hyperparameters, params, dataset):
	""" Runs the classifier with a given set of hyperparameters"""

	# Split boosts from other hyperparameters and format accordingly
	boost_vectors, boost_labels, other = split_hyperparameters(hyperparameters)

	# Override Params
	params["elasticsearch"]["boost_labels"] = boost_labels
	params["elasticsearch"]["boost_vectors"] = boost_vectors

	# Run Classifier with new Hyperparameters
	desc_queue, non_physical = get_desc_queue(dataset)
	accuracy = tokenize(params, desc_queue, other, non_physical)

	return accuracy

def run_iteration(top_score, params, known, dataset):

	hyperparameters = top_score['hyperparameters']
	new_top_score = top_score
	learning_rate = settings["iteration_learning_rate"]
	iterations = settings["iteration_search_space"]

	for i in range(iterations):
		randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=learning_rate)
		print("\n", "ITERATION NUMBER: " + str(i))
		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params, dataset)
		same_or_higher_recall = accuracy["total_recall_non_physical"] >= new_top_score["total_recall_non_physical"]
		same_or_higher_precision = accuracy["precision"] >= new_top_score["precision"]
		not_too_high_precision = accuracy["precision"] <= settings["max_precision"]

		if same_or_higher_recall and same_or_higher_precision and not_too_high_precision:
			new_top_score = accuracy
			new_top_score['hyperparameters'] = randomized_hyperparameters
			print("\n", "SCORE: " + str(accuracy["precision"]))

	return new_top_score

def gradient_ascent(initial_values, params, known, dataset):

	top_score = initial_values
	learning_rate = settings["iteration_learning_rate"]
	save_top_score(initial_values)
	iterations = settings["gradient_ascent_iterations"]

	for i in range(iterations):
		top_score = run_iteration(top_score, params, known, dataset)
		
		# Save Iterations Top Hyperparameters
		save_top_score(top_score)

	return top_score

def save_top_score(top_score):

	record = open(os.path.splitext(os.path.basename(sys.argv[1]))[0] + "_top_scores.txt", "a")
	pprint("Precision = " + str(top_score['precision']) + "%", record)
	pprint("Best Recall = " + str(top_score['total_recall_non_physical']) + "%", record)
	boost_vectors, boost_labels, other = split_hyperparameters(top_score["hyperparameters"])
	pprint(boost_vectors, record)
	pprint(other, record)
	record.close()

def split_hyperparameters(hyperparameters):
	
	boost_vectors = {}
	boost_labels = ["standard_fields"]
	other = {}

	for key, value in hyperparameters.items():
		if key == "es_result_size" or key == "z_score_threshold":
			other[key] = value
		else:
			boost_vectors[key] = [value]
		
	return boost_vectors, boost_labels, other

def randomized_optimization(hyperparameters, known, params, dataset):

	"""Generates randomized parameter keys by
	providing a range to sample from. 
	Runs the classifier a fixed number of times and
	provides the top score found"""

	# Init
	initial_values = get_initial_values(hyperparameters, params, known, dataset)

	# Run Gradient Ascent 
	top_score = gradient_ascent(initial_values, params, known, dataset)

	print("Precision = " + str(top_score['precision']) + "%")
	print("Best Recall = " + str(top_score['total_recall_non_physical']) + "%")
	print("HYPERPARAMETERS:")
	pprint(top_score["hyperparameters"])
	print("ALL RESULTS:")
	
	# Save Final Parameters
	file_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] + "_" + str(round(top_score['precision'], 2)) + "Precision" + str(round(top_score['total_recall_non_physical'], 2)) + "Recall.json" 
	new_parameters = open(file_name, 'w')
	#pprint(top_score["hyperparameters"], new_parameters)

	return top_score

def test_train_split(params):
	"""Load the verification source, and split the
	data into a test and training set"""

	verification_source = params.get("verification_source", "data/misc/verifiedLabeledTrans.csv")
	test_size = settings["test_size"]
	test, train = [], []
	dataset = []

	# Load Data
	verification_file = open(verification_source, encoding="utf-8", errors='replace')
	verified_transactions = list(csv.DictReader(verification_file))
	verification_file.close()

	for i in range(len(verified_transactions)):
		curr = verified_transactions[i]
		category = "IS_PHYSICAL_TRANSACTION"
		if curr["factual_id"] != "" or curr[category] == "0" or curr[category] == "2":
			dataset.append(curr)

	# Shuffle/Split Data
	shuffle(dataset)
	split_arr = array_split(array(dataset), 2)
	test = list(split_arr[0])
	train = list(split_arr[1])

	return test, train

def get_desc_queue(dataset):
	"""Alt version of get_desc_queue"""

	transactions = [trans["DESCRIPTION"] for trans in dataset]
	desc_queue, non_physical = queue.Queue(), []

	# Run Binary Classifier
	for transaction in transactions:
		prediction = predict_if_physical_transaction(transaction)
		if prediction == "1":
			desc_queue.put(transaction)
		elif prediction == "0":
			non_physical.append(transaction)
		elif prediction == "2":
			desc_queue.put(transaction)

	return desc_queue, non_physical

def cross_validate(top_score, dataset):
	"""Validate model built with training set against
	test set"""

	# Split boosts from other hyperparameters and format accordingly
	hyperparameters = top_score["hyperparameters"]
	boost_vectors, boost_labels, other = split_hyperparameters(hyperparameters)

	# Override Params
	params["elasticsearch"]["boost_labels"] = boost_labels
	params["elasticsearch"]["boost_vectors"] = boost_vectors

	# Run Classifier
	print("\n", "CROSS VALIDATE", "\n")
	desc_queue, non_physical = get_desc_queue(dataset)
	accuracy = tokenize(params, desc_queue, other, non_physical)

	return accuracy

def two_fold(hyperparameters, known, params, d0, d1):
	"""Run two-fold cross validation. Combine Scores"""

	# Run Randomized Optimization and Cross Validate
	d0_top_score = randomized_optimization(hyperparameters, known, params, d0)
	d1_top_score = randomized_optimization(hyperparameters, known, params, d1)

	# Cross Validate
	d0_results = cross_validate(d0_top_score, d1)
	d1_results = cross_validate(d1_top_score, d0)

	# See Scores
	print("FINAL SCORES:")
	pprint(d0_results)
	pprint(d1_results)

if __name__ == "__main__":

	# Clear Contents from Previous Runs
	open(os.path.splitext(os.path.basename(sys.argv[1]))[0] + '_top_scores.txt', 'w').close()

	settings = {
		"test_size": 0.5,
		"initial_search_space": 50,
		"initial_learning_rate": 1,
		"iteration_search_space": 35,
		"iteration_learning_rate": 0.3,
		"gradient_ascent_iterations": 15,
		"max_precision": 96
	}

	known = {"es_result_size" : "45"}

	hyperparameters = {    
	    "name" : "3",             
	    "address" : "1",          
	    "address_extended" : "1", 
	    "po_box" : "1",           
	    "locality" : "1",         
	    "region" : "1",           
	    "post_town" : "1",        
	    "admin_region" : "1",     
	    "postcode" : "1",                
	    "tel" : "1",                            
	    "neighborhood" : "1",     
	    "email" : "1",               
	    "category_labels" : "1",           
	    "chain_name" : "1",
		"z_score_threshold" : "3"
	}

	# Meta Information
	start_time = datetime.datetime.now()
	params = initialize()
	test, train = test_train_split(params)

	# Run Two Fold Cross Validation
	two_fold(hyperparameters, known, params, test, train)

	# Run Speed Tests
	time_delta = datetime.datetime.now() - start_time
	print("TOTAL TIME TAKEN FOR OPTIMIZATION: ", time_delta)
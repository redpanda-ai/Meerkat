#!/usr/local/bin/python3.3
# pylint: disable=all

"""This module is the core of the Meerkat
engine. It allows us to rapidly evaluate many
possible configurations if provided a well
labeled dataset. Iteratively it runs Meerkat
with randomized levels of configurations and
then converges on the best possible values. 

In context of Machine Leaning, this module 
performs hyperparameter optimization. This 
involves tuning the numeric values located at  
Meerkat/config/hyperparameters. These key 
value pairs map to hyperparameters used through 
out the Meerkat Classifier in aid of tuning or
refining the queries used with ElasticSearch.

This module utilizes a common method known as 
grid search. In particular we are using a custom
implementation of randomized optimization as it 
works better where it is resource intensive to 
exaustively perform a standard grid_search.

Created on Feb 26, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 -m meerkat.optimization config/train.json
# Note: Experts only! Do not touch!

#FEATURE_IDEA: Memory of each round of configuration
#FEATURE: limit size of user data. One thread taking way longer. 
#FEATURE: Run initial values the first time for a benchmark
#FEATURE: Save every top score found from "get initial values"

#####################################################

import sys 
import datetime
import os
import queue
import csv
import collections
from copy import deepcopy
from random import randint, uniform, random, shuffle

from pprint import pprint
from numpy import array, array_split

from meerkat.description_consumer import DescriptionConsumer
from meerkat.binary_classifier.load import select_model
from meerkat.accuracy import print_results, test_accuracy
from meerkat.various_tools import load_dict_list, queue_to_list, load_params, load_hyperparameters

def test_train_split(dataset):
	"""Load the verification source, and split the
	data into a test and training set"""

	# Shuffle/Split Data
	test, train = [], []
	shuffle(dataset)
	split_arr = array_split(array(dataset), 2)
	test = list(split_arr[0])
	train = list(split_arr[1])

	return test, train

def two_fold(hyperparameters, known, params):
	"""Run two-fold cross validation. Combine Scores"""

	# Run Randomized Optimization and Cross Validate
	d0_top_score = randomized_optimization(hyperparameters, known, params, d0)
	d1_top_score = randomized_optimization(hyperparameters, known, params, d1)

	# Cross Validate
	d1_results = cross_validate(d0_top_score, d1)
	d0_results = cross_validate(d1_top_score, d0)

	# Show Scores
	print("FINAL SCORES:")
	pprint(d0_results)
	pprint(d1_results)

	# Save Scores
	save_cross_fold_results(d0_top_score, d0_results, d1_top_score, d1_results)

def cross_validate(top_score, dataset):
	"""Validate model built with training set against
	test set"""

	# Split boosts from other hyperparameters and format accordingly
	hyperparameters = top_score["hyperparameters"]
	boost_vectors, boost_labels, hyperparameters = split_hyperparameters(hyperparameters)

	# Override Params
	params["elasticsearch"]["boost_labels"] = boost_labels
	params["elasticsearch"]["boost_vectors"] = boost_vectors

	# Run Classifier
	print("\n", "CROSS VALIDATE", "\n")
	desc_queue = get_desc_queue(dataset)
	accuracy = run_meerkat(params, desc_queue, hyperparameters)

	return accuracy

def save_cross_fold_results(d0_top_score, d0_results, d1_top_score, d1_results):

	# Save Scores
	file_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] + "_" + "cross_validation_results.json" 
	record = open(file_name, 'w')

	pprint("d0 as Training Data - Top Score found through Optimization:", record)
	pprint(d0_top_score, record)

	pprint("d0 as Training Data - Score of d1 on this set of hyperparameters:", record)
	pprint(d1_results, record)

	pprint("d1 as Training Data - Top Score found through Optimization:", record)
	pprint(d1_top_score, record)

	pprint("d1 as Training Data - Score of d0 on this set of hyperparameters:", record)
	pprint(d0_results, record)

def run_meerkat(params, desc_queue, hyperparameters):
	"""Run meerkat on a set of transactions"""

	consumer_threads = params.get("concurrency", 8)
	result_queue = queue.Queue()

	for i in range(consumer_threads):
		new_consumer = DescriptionConsumer(i, params, desc_queue,\
			result_queue, hyperparameters)
		new_consumer.setDaemon(True)
		new_consumer.start()

	desc_queue.join()

	# Convert queue to list
	result_list = queue_to_list(result_queue)

	# Test Accuracy
	accuracy_results = test_accuracy(params, result_list=result_list)
	print_results(accuracy_results)

	return accuracy_results

def load_dataset(params):
	"""Load a verified dataset"""

	verification_source = params.get("verification_source", "data/misc/ground_truth_card.txt")
	dataset = []

	# Load Data
	verified_transactions = load_dict_list(verification_source)

	# Filter Verification File
	for i in range(len(verified_transactions)):
		curr = verified_transactions[i]
		if curr["factual_id"] != "":
			dataset.append(curr)

	return dataset

def randomized_optimization(hyperparameters, known, params, dataset):

	"""Generates randomized parameter keys by
	providing a range to sample from. 
	Runs the classifier a fixed number of times and
	provides the top score found"""

	# Init
	base_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
	initial_values = get_initial_values(hyperparameters, params, known, dataset)

	# Run Gradient Ascent 
	top_score = gradient_descent(initial_values, params, known, dataset)

	# Save All Scores
	with open(base_name + "_all_scores.txt", "w") as fout:
		pprint(params["scores"], fout)

	print("Precision = " + str(top_score['precision']) + "%")
	print("Best Recall = " + str(top_score['total_recall_physical']) + "%")
	print("HYPERPARAMETERS:")
	pprint(top_score["hyperparameters"])
	print("ALL RESULTS:")
	
	# Save Final Parameters
	file_name = base_name + "_" + str(round(top_score['precision'], 2)) + "Precision" + str(round(top_score['total_recall_physical'], 2)) + "Recall.json" 
	new_parameters = open(file_name, 'w')
	pprint(top_score["hyperparameters"], new_parameters)

	return top_score

def get_initial_values(hyperparameters, params, known, dataset):
	"""Do a simple search to find starter values"""

	settings = params["settings"]
	top_score = {"precision" : 0, "total_recall_physical" : 0, "f1" : 0}
	iterations = settings["initial_search_space"]
	learning_rate = settings["initial_learning_rate"]

	print("Training on " + str(len(dataset)) + " unique transactions")

	for i in range(iterations):

		randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=learning_rate)
		print("\n", "ITERATION NUMBER: " + str(i))
		print("\n", randomized_hyperparameters,"\n")

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params, dataset)
		precision = accuracy["precision"]
		recall = accuracy["total_recall_physical"]
		f1 = 2 * (((precision / 100) * 1.3 * recall) / (((precision / 100) * 1.3) + recall))
		higher_f1 = f1 >= top_score["f1"]
		higher_precision = precision >= top_score["precision"]
		higher_recall = recall >= top_score["total_recall_physical"]
		not_too_high = precision <= settings["max_precision"]

		if higher_recall:
			top_score = accuracy
			top_score["f1"] = f1
			top_score['hyperparameters'] = randomized_hyperparameters
			print("\n", "SCORE PRECISION: " + str(round(accuracy["precision"], 2)))
			print("\n", "SCORE RECALL: " + str(round(accuracy["total_recall_physical"], 2)))

		# Keep Track of All Scores
		score = {
			"hyperparameters" : randomized_hyperparameters,
			"precision" : round(accuracy["precision"], 2),
			"recall" : round(accuracy["total_recall_physical"], 2)
		}

		params["scores"].append(score)
		
		print("\n", randomized_hyperparameters,"\n")

	print("TOP SCORE:" + str(top_score["precision"]))
	return top_score

def gradient_descent(initial_values, params, known, dataset):

	settings = params["settings"]
	top_score = initial_values
	learning_rate = settings["iteration_learning_rate"]
	save_top_score(initial_values)
	iterations = settings["gradient_descent_iterations"]

	for i in range(iterations):
		top_score = run_iteration(top_score, params, known, dataset)
		
		# Save Iterations Top Hyperparameters
		save_top_score(top_score)

	return top_score

def run_iteration(top_score, params, known, dataset):

	settings = params["settings"]
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
		same_or_higher_recall = accuracy["total_recall_physical"] >= new_top_score["total_recall_physical"]
		same_or_higher_precision = accuracy["precision"] >= new_top_score["precision"]
		not_too_high_precision = accuracy["precision"] <= settings["max_precision"]

		if same_or_higher_recall and same_or_higher_precision and not_too_high_precision:
			new_top_score = accuracy
			new_top_score['hyperparameters'] = randomized_hyperparameters
			print("\n", "SCORE PRECISION: " + str(round(accuracy["precision"], 2)))
			print("\n", "SCORE RECALL: " + str(round(accuracy["total_recall_physical"], 2)))

		score = {
			"hyperparameters" : randomized_hyperparameters,
			"precision" : round(accuracy["precision"], 2),
			"recall" : round(accuracy["total_recall_physical"], 2)
		}

		params["scores"].append(score)

	return new_top_score

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

def save_top_score(top_score):

	record = open(os.path.splitext(os.path.basename(sys.argv[1]))[0] + "_top_scores.txt", "a")
	pprint("Precision = " + str(top_score['precision']) + "%", record)
	pprint("Best Recall = " + str(top_score['total_recall_physical']) + "%", record)
	boost_vectors, boost_labels, other = split_hyperparameters(top_score["hyperparameters"])
	pprint(boost_vectors, record)
	pprint(other, record)
	record.close()

def split_hyperparameters(hyperparameters):
	
	boost_vectors = {}
	boost_labels = ["standard_fields"]
	non_boost = ["es_result_size", "z_score_threshold", "good_description"]
	other = {}

	for key, value in hyperparameters.items():
		if key in non_boost:
			other[key] = value
		else:
			boost_vectors[key] = [value]
		
	return boost_vectors, boost_labels, other

def get_desc_queue(dataset):
	"""Alt version of get_desc_queue"""

	transactions = [deepcopy(X) for X in dataset]
	desc_queue = queue.Queue()
	users = collections.defaultdict(list)

	# Split into user buckets
	for row in transactions:
		user = row['UNIQUE_MEM_ID']
		users[user].append(row)

	# Add Users to Queue
	for key, value in users.items():
		desc_queue.put(users[key])

	return desc_queue

def run_classifier(hyperparameters, params, dataset):
	""" Runs the classifier with a given set of hyperparameters"""

	# Split boosts from other hyperparameters and format accordingly
	boost_vectors, boost_labels, hyperparameters = split_hyperparameters(hyperparameters)

	# Override Params
	params["elasticsearch"]["boost_labels"] = boost_labels
	params["elasticsearch"]["boost_vectors"] = boost_vectors

	# Run Classifier with new Hyperparameters
	desc_queue = get_desc_queue(dataset)
	accuracy = run_meerkat(params, desc_queue, hyperparameters)

	return accuracy

def verify_arguments():
	"""Verify Usage"""

	# Must Provide Config File
	if len(sys.argv) != 2:
		print("Please provide a config file")
		sys.exit()

	# Clear Contents from Previous Runs
	open(os.path.splitext(os.path.basename(sys.argv[1]))[0] + '_top_scores.txt', 'w').close()

if __name__ == "__main__":

	# Meta Information
	start_time = datetime.datetime.now()
	verify_arguments()
	params = load_params(sys.argv[1])

	params["settings"] = {
		"folds": 1,
		"initial_search_space": 100,
		"initial_learning_rate": 3,
		"iteration_search_space": 5,
		"iteration_learning_rate": 0.5,
		"gradient_descent_iterations": 10,
		"max_precision": 95
	}

	params["scores"] = []

	known = {"es_result_size" : "45"}

	hyperparameters = {
		"internal_store_number" : "2",  
	    "name" : "3",
	    "good_description" : "2",             
	    "address" : "0.5",          
	    "address_extended" : "1.25", 
	    "po_box" : "1.25",           
	    "locality" : "1.5",         
	    "region" : "1.5",           
	    "post_town" : "0.5",        
	    "admin_region" : "1",     
	    "postcode" : "1",                
	    "tel" : "0.5",                            
	    "neighborhood" : "1",     
	    "email" : "0.5",               
	    "category_labels" : "1.25",           
	    "chain_name" : "1",
		"z_score_threshold" : "3"
	}

	dataset = load_dataset(params)

	# Use all data or Cross Validate
	if params["settings"]["folds"] == 1:
		randomized_optimization(hyperparameters, known, params, dataset)
	else:
		d0, d1 = test_train_split(dataset)
		two_fold(hyperparameters, known, params)

	# Run Speed Tests
	time_delta = datetime.datetime.now() - start_time
	print("TOTAL TIME TAKEN FOR OPTIMIZATION: ", time_delta)

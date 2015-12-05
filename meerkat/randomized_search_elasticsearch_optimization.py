#!/usr/local/bin/python3.3
# pylint: disable=line-too-long

"""This module is the core of the Meerkat engine. It allows us to rapidly
evaluate many possible configurations if provided a well labeled dataset.
Iteratively it runs Meerkat with randomized levels of configurations and
then converges on the best possible values.

In context of Machine Leaning, this module  performs hyperparameter
optimization. This involves tuning the numeric values located at
Meerkat/config/hyperparameters. These key value pairs map to
hyperparameters used through out the Meerkat Classifier in aid of tuning
or refining the queries used with ElasticSearch.

This module utilizes a common method known as grid search. In particular
we are using a custom implementation of randomized optimization as it
works better where it is resource intensive to exaustively perform a
standard grid_search.

Created on Feb 26, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 -m meerkat.randomized_search_elasticsearch_optimization config/train.json
# Note: Experts only! Do not touch!

#####################################################

import sys
import datetime
import os
import json
from random import uniform

from pprint import pprint

from meerkat.accuracy import print_results, vest_accuracy
from meerkat.various_tools import load_dict_list, safe_print, get_us_cities
from meerkat.various_tools import load_params
from meerkat.web_service.web_consumer import WebConsumer

PARAMS = load_params(sys.argv[1])
CITIES = get_us_cities()
consumer = WebConsumer(cities=CITIES, params=PARAMS)
BATCH_SIZE = 100

def run_meerkat(params, dataset):
	"""Run meerkat on a set of transactions"""

	result_list = []
	number = (len(dataset))/BATCH_SIZE
	number = int(number - (number%1))
	for x in range(0, number+1):
		batch = []
		for i in range(x*BATCH_SIZE, (x+1)*BATCH_SIZE):
			try:
				batch.append(dataset[i])
			except IndexError:
				break

		print("Batch number: {0}".format(x))
		batch_in = format_web_consumer(batch)
		batch_result = consumer.classify(batch_in, optimizing=True)
		result_list.extend(batch_result["transaction_list"])

	# Test Accuracy
	#pprint(result_list)
	accuracy_results = vest_accuracy(params, result_list=result_list)
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

		for field in params["output"]["results"]["labels"]:
			curr[field] = ""

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
	with open("optimization_results/" + base_name + "_all_scores.txt", "w") as fout:
		pprint(params["optimization"]["scores"], fout)

	print("Precision = " + str(top_score['precision']) + "%")
	print("Best Recall = " + str(top_score['total_recall_physical']) + "%")
	print("HYPERPARAMETERS:")
	print("ALL RESULTS:")

	# Save Final Parameters
	str_precision = str(round(top_score['precision'], 2))
	str_recall = str(round(top_score['total_recall_physical'], 2))
	file_name = "optimization_results/" + base_name + "_" + str_precision + "Precision" + str_recall + "Recall.json"
	new_parameters = open(file_name, 'w')
	pprint(top_score["hyperparameters"], new_parameters)

	return top_score

def display_hyperparameters(hyperparameters):
	"""Display a human readable output of hyperparameters"""

	safe_print("Iteration Hyperparameters:\n")
	for key, value in hyperparameters.items():
		safe_print("{:29} : {}".format(key, value))

	sys.stdout.write("\n")

def get_initial_values(hyperparameters, params, known, dataset):
	"""Do a simple search to find starter values"""

	settings = params["optimization"]["settings"]
	top_score = {"precision" : 0, "total_recall_physical" : 0}

	print("Training on " + str(len(dataset)) + " unique transactions")

	for i in range(settings["initial_search_space"]):

		if i > 0:
			randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=settings["initial_learning_rate"])
		else:
			randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=0)

		safe_print("\n--------------------\n")
		safe_print("ITERATION NUMBER: " + str(i) + "\n")
		display_hyperparameters(randomized_hyperparameters)

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params, dataset)
		precision = accuracy["precision"]
		recall = accuracy["total_recall_physical"]
		same_or_higher_precision = precision >= top_score["precision"]
		not_too_high = precision <= settings["max_precision"]
		has_min_recall = recall >= settings["min_recall"]

		if has_min_recall and not_too_high and same_or_higher_precision:
			top_score = accuracy
			top_score['hyperparameters'] = randomized_hyperparameters
			safe_print("\nSCORE PRECISION: " + str(round(accuracy["precision"], 2)))
			safe_print("SCORE RECALL: " + str(round(accuracy["total_recall_physical"], 2)) + "\n")

		# Keep Track of All Scores
		score = {
			"hyperparameters" : randomized_hyperparameters,
			"precision" : round(accuracy["precision"], 2),
			"recall" : round(accuracy["total_recall_physical"], 2)
		}

		params["optimization"]["scores"].append(score)

		display_hyperparameters(randomized_hyperparameters)

	print("TOP SCORE:" + str(top_score["precision"]))
	return top_score

def gradient_descent(initial_values, params, known, dataset):

	settings = params["optimization"]["settings"]
	top_score = initial_values
	save_top_score(initial_values)
	iterations = settings["gradient_descent_iterations"]

	print("\n----------- Stochastic Gradient Descent ------------")

	for _ in range(iterations):
		top_score = run_iteration(top_score, params, known, dataset)

		# Save Iterations Top Hyperparameters
		print("\n", "Top Precision: " + str(round(top_score["precision"], 2)))
		print("\n", "Top Recall: " + str(round(top_score["total_recall_physical"], 2)))
		save_top_score(top_score)

	return top_score

def run_iteration(top_score, params, known, dataset):

	settings = params["optimization"]["settings"]
	hyperparameters = top_score['hyperparameters']
	new_top_score = top_score
	learning_rate = settings["iteration_learning_rate"]

	for i in range(settings["iteration_search_space"]):

		randomized_hyperparameters = randomize(hyperparameters, known, learning_rate=learning_rate)

		# Iteration Stats
		safe_print("\n--------------------\n")
		safe_print("ITERATION NUMBER: " + str(i) + "\n")
		display_hyperparameters(randomized_hyperparameters)

		# Run Classifier
		accuracy = run_classifier(randomized_hyperparameters, params, dataset)
		same_or_higher_recall = accuracy["total_recall_physical"] >= new_top_score["total_recall_physical"]
		same_or_higher_precision = accuracy["precision"] >= new_top_score["precision"]
		not_too_high_precision = accuracy["precision"] <= settings["max_precision"]

		if same_or_higher_recall and same_or_higher_precision and not_too_high_precision:

			new_top_score = accuracy
			new_top_score['hyperparameters'] = randomized_hyperparameters
			safe_print("\n", "SCORE PRECISION: " + str(round(accuracy["precision"], 2)))
			safe_print("\n", "SCORE RECALL: " + str(round(accuracy["total_recall_physical"], 2)))

		score = {
			"hyperparameters" : randomized_hyperparameters,
			"precision" : round(accuracy["precision"], 2),
			"recall" : round(accuracy["total_recall_physical"], 2)
		}

		params["optimization"]["scores"].append(score)

	return new_top_score

def randomize(hyperparameters, known={}, learning_rate=0.3):
	"""Finds new values within a given range
	based on a provided learning rate"""

	randomized = {}

	for key, value in hyperparameters.items():
		lower = float(value) - learning_rate
		lower = lower if lower >= 0 else 0
		upper = float(value) + learning_rate
		upper = upper if upper <= 3 else 3
		new_value = uniform(lower, upper)

		# HACK
		if key == "z_score_threshold":
			lower_upper_bound = float(value) + (learning_rate / 2)
			new_value = uniform(lower, lower_upper_bound)

		randomized[key] = str(round(new_value, 3))

	return dict(list(randomized.items()) + list(known.items()))

def save_top_score(top_score):

	record = open("optimization_results/" + os.path.splitext(os.path.basename(sys.argv[1]))[0] +
		"_top_scores.txt", "a")
	pprint("Precision = " + str(top_score['precision']) + "%", record)
	pprint("Best Recall = " + str(top_score['total_recall_physical']) +"%", record)
	boost_vectors, _, other = split_hyperparameters(top_score["hyperparameters"])
	pprint(boost_vectors, record)
	pprint(other, record)
	record.close()

def split_hyperparameters(hyperparameters):
	"""partition hyperparameters into 2 parts based on keys and non_boost list"""
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

def run_classifier(hyperparameters, params, dataset):
	""" Runs the classifier with a given set of hyperparameters"""

	# Split boosts from other hyperparameters and format accordingly
	boost_vectors, boost_labels, hyperparameters = split_hyperparameters(hyperparameters)

	# Override Params
	hyperparameters["boost_labels"] = boost_labels
	hyperparameters["boost_vectors"] = boost_vectors

	consumer.update_hyperparams(hyperparameters)
	accuracy = run_meerkat(params, dataset)

	return accuracy

def add_local_params(params):
	"""Adds additional local params"""

	params["mode"] = "train"
	params["optimization"] = {}
	params["optimization"]["scores"] = []

	params["optimization"]["settings"] = {
		"initial_search_space": 25,
		"initial_learning_rate": 0.25,
		"iteration_search_space": 15,
		"iteration_learning_rate": 0.1,
		"gradient_descent_iterations": 10,
		"max_precision": 97.5,
		"min_recall": 31
	}

	return params

def verify_arguments():
	"""Verify Usage"""

	# Must Provide Config File
	if len(sys.argv) != 2:
		print("Please provide a config file")
		sys.exit()

	# Clear Contents from Previous Runs
	previous_scores = "optimization_results/" + os.path.splitext(os.path.basename(sys.argv[1]))[0] + "_top_scores.txt"
	if os.path.isfile(previous_scores):
		open(previous_scores, 'w').close()

def format_web_consumer(dataset):

	formatted = json.load(open("meerkat/web_service/example_input.json", "r"))
	formatted["transaction_list"] = dataset
	trans_id = 1
	for trans in formatted["transaction_list"]:
		trans["transaction_id"] = trans_id
		trans_id = trans_id +1
		trans["description"] = trans["DESCRIPTION_UNMASKED"]
		trans["amount"] = trans["AMOUNT"]
		trans["date"] = trans["TRANSACTION_DATE"]
		trans["ledger_entry"] = "credit"

	return formatted

def run_from_command_line():
	"""Runs these commands if the module is invoked from the command line"""

	# Meta Information
	start_time = datetime.datetime.now()
	verify_arguments()
	# Add Local Params
	params = add_local_params(PARAMS)

	known = {
		"es_result_size" : "45"
	}

	hyperparameters = {
		"address" : "0",
	    "address_extended" : "0",
	    "locality" : "1.367",
	    "region" : "1.685",
	    "post_town" : "0.577",
	    "admin_region" : "0.69",
	    "postcode" : "0.9",
	    "tel" : "0.6",
	    "neighborhood" : "0.801",
	    "email" : "0.5",
	    "category_labels" : "1.319",
	    "chain_name" : "1",
	    "internal_store_number" : "1.9",
	    "name" : "2.781",
	    "good_description" : "2",
		"z_score_threshold" : "2.857",
		"po_box" : "1.159",
	}

	# Use all data and optimize
	dataset = load_dataset(params)
	randomized_optimization(hyperparameters, known, params, dataset)

	# Run Speed Tests
	time_delta = datetime.datetime.now() - start_time
	print("TOTAL TIME TAKEN FOR OPTIMIZATION: ", time_delta)

if __name__ == "__main__":
	run_from_command_line()

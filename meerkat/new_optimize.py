#!/usr/local/bin/python3.3
# pylint: disable=all

"""This module is the core of the Meerkat engine. It allows us to rapidly
evaluate many possible configurations if provided a well labeled dataset.
Iteratively it runs Meerkat with randomized levels of configurations and
then converges on the best possible values. 

In context of Machine Learning, this module performs hyperparameter
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

# python3.3 -m meerkat.new_optimize config/train.json
# Note: Experts only! Do not touch!

#####################################################

import sys 
import datetime
import os
import queue
import csv
import collections
import contextlib
from copy import deepcopy
from random import randint, uniform, random, shuffle

from pprint import pprint
import numpy as np
from numpy import array, array_split
from scipy.optimize import minimize, brute, basinhopping

from meerkat.consumer import Consumer
from meerkat.binary_classifier.load import select_model
from meerkat.accuracy import print_results, test_accuracy
from meerkat.various_tools import load_dict_list, queue_to_list, safe_print, get_us_cities
from meerkat.various_tools import load_params, load_hyperparameters, progress

CITIES = get_us_cities()

class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:
            # this could be done in a much more clever way, but it will work for example purposes
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break

        return xnew

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = DummyFile()
    sys.stderr = DummyFile()
    yield
    sys.stderr = save_stderr
    sys.stdout = save_stdout

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

def run_meerkat(params, desc_queue, hyperparameters):
	"""Run meerkat on a set of transactions"""

	consumer_threads = params.get("concurrency", 8)
	result_queue = queue.Queue()

	# Suppress Output and Classify
	for i in range(consumer_threads):
		new_consumer = Consumer(i, params, desc_queue, result_queue, hyperparameters, CITIES)
		new_consumer.setDaemon(True)
		new_consumer.start()

	# Progress 
	qsize = desc_queue.qsize()
	total = range(qsize)

	while qsize > 0:
		if qsize == desc_queue.qsize():
			continue
		else:
			qsize = desc_queue.qsize()
			if params["mode"] == "train":
				progress((len(total) - qsize), total, message="complete with current iteration")

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

		for field in params["output"]["results"]["labels"]:
			curr[field] = ""

		dataset.append(curr)

	return dataset

def save_top_score(top_score):

	record = open("optimization_results/" + os.path.splitext(os.path.basename(sys.argv[1]))[0] + "_top_scores.txt", "a")
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
	hyperparameters["boost_labels"] = boost_labels
	hyperparameters["boost_vectors"] = boost_vectors

	# Run Classifier with new Hyperparameters. Suppress Output
	desc_queue = get_desc_queue(dataset)
	accuracy = run_meerkat(params, desc_queue, hyperparameters)

	return accuracy

def add_local_params(params):
	"""Adds additional local params"""

	params["mode"] = "train"
	params["optimization"] = {}
	params["optimization"]["scores"] = []

	params["optimization"]["settings"] = {
		"folds": 1,
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
	open("optimization_results/" + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '_top_scores.txt', 'w').close()

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""

	# Meta Information
	verify_arguments()
	params = load_params(sys.argv[1])

	# Add Local Params
	add_local_params(params)

	hyperparams = {
		"address" : "0",          
	    "address_extended" : "0",
	    "admin_region" : "0.69",
	    "category_labels" : "1.319",          
	    "locality" : "1.367",         
	    "region" : "1.685",           
	    "post_town" : "0.577",             
	    "postcode" : "0.9",                
	    "tel" : "0.6",                            
	    "neighborhood" : "0.801",     
	    "email" : "0.5",                      
	    "chain_name" : "1",
	    "internal_store_number" : "1.9",  
	    "name" : "2.781",
	    "good_description" : "2",
		"z_score_threshold" : "2.897",
		"po_box" : "1.159",
	}

	hyperparams = {
		"address" : "1",          
	    "address_extended" : "1",
	    "admin_region" : "1",
	    "category_labels" : "1",          
	    "locality" : "1",         
	    "region" : "1",           
	    "post_town" : "1",             
	    "postcode" : "1",                
	    "tel" : "1",                            
	    "neighborhood" : "1",     
	    "email" : "1",                      
	    "chain_name" : "1",
	    "internal_store_number" : "1",  
	    "name" : "1",
	    "good_description" : "1",
		"z_score_threshold" : "2.9",
		"po_box" : "1",
	}

	param_names = list(hyperparams.keys())
	initial_guess = [float(x) for x in list(hyperparams.values())]
	x0 = np.array(initial_guess)
	bounds = [((45, 45) if x == "es_result_size" else (0, 3)) for x in param_names]
	xmin = np.array([x[0] for x in bounds])
	xmax = np.array([x[1] for x in bounds])
	dataset = load_dataset(params)

	safe_print(param_names)

	def loss(x):
		"""A loss function to optimize against"""

		x = take_step(x)
		x = [str(n) for n in x]
		hyperparam = dict(zip(param_names, list(x)))
		hyperparam['es_result_size'] = "45"
		accuracy = run_classifier(hyperparam, params, dataset)
		error = (100 - accuracy['precision']) / 100

		safe_print(error)

		return error

	take_step = RandomDisplacementBounds(xmin, xmax, 0.3)
	minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, options={"maxiter": 10, "disp": True})
	res = basinhopping(loss, x0, niter=50, minimizer_kwargs=minimizer_kwargs, take_step=take_step, niter_success=5)
	safe_print(res)

if __name__ == "__main__":
	run_from_command_line(sys.argv)

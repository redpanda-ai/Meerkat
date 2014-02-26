#!/usr/local/bin/python3
# pylint: disable=C0301

"""This module performs hyperparameter optimization. This involves
tuning the keys located under Longtail/config/keys. Those key value
pairs map to hyperparameters used through out the Longtail Classifier.
This module utilizes a common method known as grid search. In particular
we are using randomized optimization as it works better where it
is resource intensive to exaustively perform a standard grid_search"""

from longtail.description_producer import initialize, get_desc_queue, tokenize, load_parameter_key

from time import time
from scipy.stats import randint, uniform
from sklearn.grid_search import RandomizedSearchCV

def randomized_optimization():

	"""Generates randomized parameter keys by
	providing a range and distribution to sample from. 
	Runs the classifier a fixed number of times and
	provides the top score found"""

	#Runs the entire program.
	params = initialize()
	results = []

	# Run 20 times
	for i in range(25):

		desc_queue, non_physical = get_desc_queue(params)

		# specify parameters and distributions to sample from
		randomized_hyperparameters = {
			"es_result_size" : str(randint(15, 45).rvs()),
			"z_score_threshold" : str(round(uniform(1, 2).rvs(), 2)),
			"business_name_boost" : str(round(uniform(0, 1).rvs(), 2)),
			"address_boost" : str(round(uniform(0, 1).rvs(), 2)),
			"phone_boost" : str(round(uniform(0, 1).rvs(), 2))
		}

		print(randomized_hyperparameters)

		accuracy = tokenize(params, desc_queue, randomized_hyperparameters, non_physical)

		if accuracy['precision'] > 95:
			accuracy['hyperparameters'] = randomized_hyperparameters
			results.append(accuracy)

	top_score = {"total_recall":0}

	for score in results:
		if score["total_recall"] > top_score["total_recall"]:
			top_score = score

	print("Precision = " + str(top_score['precision']) + "%")
	print("Best Recall = " + str(top_score['total_recall']) + "%")
	print("HYPERPARAMETERS:")
	print(top_score["hyperparameters"])

if __name__ == "__main__":
	randomized_optimization()
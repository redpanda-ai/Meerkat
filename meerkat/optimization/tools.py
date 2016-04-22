"""A library of useful functions for our optimization package"""
import logging
import json

from meerkat.various_tools import load_dict_list
\
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
		"max_predictive_accuracy": 97.5,
		"min_recall": 31,
		"min_percent_labeled": 31
	}
	return params

def format_web_consumer(dataset):
	"""Format the input json file """
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

def load_dataset(params):
	"""Load a verified dataset"""
	verification_source = \
	params.get("verification_source", "data/misc/ground_truth_card.txt")
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

if __name__ == "__main__":
	logging.critical("This is a library of useful functions, do not run it from the command line.")




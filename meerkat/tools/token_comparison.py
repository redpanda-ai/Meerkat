#!/usr/local/bin/python3.3

"""This script collects the most common tokens in a provided transaction
set that are either not found in Factual data, or found significantly less
often. This information is then used with the synonyms function found in
various_tools to patch differences between datasets.

Created on May 21, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Currently in Development. Experts only!

#####################################################

import random
#import sys
from collections import OrderedDict

import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from pprint import pprint

from meerkat.various_tools import load_dict_list, string_cleanse

def random_sample(corpus, sample_size):
	"""Takes a bag of words approach to vectorizing our text corpus"""

	rand_index = random.sample(range(len(corpus)), sample_size)
	content = [corpus[i] for i in rand_index]

	return content

def vectorize(corpus):
	"""Takes a bag of words approach to vectorizing our text corpus"""
	vectorizer = CountVectorizer(min_df=50, ngram_range=(1, 1))
	count_vector = vectorizer.fit_transform(corpus).toarray()
	num_samples, _ = count_vector.shape
	vocab = vectorizer.get_feature_names()

	#term_weighting(vocab, count_vector, corpus)
	distribution_dict = token_count(vocab, count_vector, num_samples)

	return distribution_dict

def token_count(vocab, count_vector, num_samples):
	"""Please supply a better docstring"""
	numpy.clip(count_vector, 0, 1, out=count_vector)
	dist = numpy.sum(count_vector, axis=0)
	dist = dist.tolist()

	#print("TOKEN COUNT:")
	#print(dict(zip(vocab, dist)), "\n")

	distribution_dict = token_frequency(vocab, dist, num_samples)

	return distribution_dict

def token_frequency(vocab, dist, num_samples):
	"""Please supply a better docstring"""
	dist[:] = [x / num_samples for x in dist]
	dist = numpy.around(dist, decimals=5).tolist()
	distribution_dict = dict(zip(vocab, dist))

	print("TOKEN FREQUENCY:")
	print(distribution_dict, "\n")

	return distribution_dict

def term_weighting(vocab, count_vector, corpus):
	"""Takes a bag of words approach to vectorizing our text corpus"""
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(count_vector)
	weighted_transaction_features = tfidf.toarray().tolist()[15]

	print(corpus[15], "\n")
	print("TOKEN IMPORTANCE SINGLE TRANSACTION. NORMALIZED "\
		"TO HAVE EUCLIDEAN NORM:")
	print(dict(zip(vocab, weighted_transaction_features)), "\n")

	print("WEIGHTS PER TOKEN:")
	print(dict(zip(vocab, numpy.around(transformer.idf_,\
		decimals=5).tolist())), "\n")

def compare_tokens(dataset_a, dataset_b):
	"""Compares available tokens"""

	deltas = {}
	print(type(dataset_a))

	for key in dataset_a:
		a_frequency = dataset_a[key]
		b_frequency = dataset_b.get(key, 0)
		deltas[key] = a_frequency - b_frequency

	print(OrderedDict(sorted(deltas.items(), key=lambda t: t[1])))

	#print(deltas)

def run_from_command_line():
	"""Runs these commands if the module is invoked from the command line"""
	# Load Files from Datasets
	factual_merchants = load_dict_list(\
		"/mnt/ephemeral/Factual_1.5_Million.txt", delimiter="\t")
	factual_merchants = [string_cleanse(merchant["address"]\
		+ " " + merchant["name"] + " " + merchant["region"]\
		+ " " + merchant["locality"]) for merchant in factual_merchants]
	transactions = load_dict_list(\
		"/mnt/ephemeral/20140415_YODLEE_CARD_PANEL_UNMASKED_physical.txt")
	transactions = [string_cleanse(transaction["DESCRIPTION"])\
		for transaction in transactions]

	# TODO Strip numbers

	# Sub Sample
	factual_merchants = random_sample(factual_merchants, 25000)
	transactions = random_sample(transactions, 25000)

	# Run Analysis
	factual_tokens = vectorize(factual_merchants)
	yodlee_tokens = vectorize(transactions)

	compare_tokens(yodlee_tokens, factual_tokens)

if __name__ == "__main__":
	run_from_command_line()


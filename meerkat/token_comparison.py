#!/usr/local/bin/python3

"""This script collects the most common tokens in our corpus that 
are either not found in factual data, or found significantly less often"""

import random, numpy, sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from meerkat.various_tools import load_dict_list, string_cleanse
from pprint import pprint
from collections import OrderedDict

def random_sample(corpus, sample_size):
	"""Takes a bag of words approach to vectorizing our text corpus"""

	randIndex = random.sample(range(len(corpus)), sample_size)
	content = [corpus[i] for i in randIndex]

	return content

def vectorize(corpus):
	"""Takes a bag of words approach to vectorizing our text corpus"""
	vectorizer = CountVectorizer(min_df=0.005, max_df=0.95, ngram_range=(1,1))
	countVector = vectorizer.fit_transform(corpus).toarray()
	num_samples, num_features = countVector.shape
	vocab = vectorizer.get_feature_names()

	#termWeighting(vocab, countVector, corpus)
	distribution_dict = tokenCount(vocab, countVector, num_samples)

	return distribution_dict

def tokenCount(vocab, countVector, num_samples):

	numpy.clip(countVector, 0, 1, out=countVector)
	dist = numpy.sum(countVector, axis=0)
	dist = dist.tolist()

	#print("TOKEN COUNT:")
	#print(dict(zip(vocab, dist)), "\n")

	distribution_dict = tokenFrequency(vocab, dist, num_samples)

	return distribution_dict

def tokenFrequency(vocab, dist, num_samples):

	dist[:] = [x / num_samples for x in dist]
	dist = numpy.around(dist, decimals=5).tolist()
	distribution_dict = dict(zip(vocab, dist))

	print("TOKEN FREQUENCY:")
	print(distribution_dict, "\n")

	return distribution_dict

def termWeighting(vocab, countVector, corpus):
	"""Takes a bag of words approach to vectorizing our text corpus"""
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(countVector)
	weighted_transaction_features = tfidf.toarray().tolist()[15]

	print(corpus[15], "\n")
	print("TOKEN IMPORTANCE SINGLE TRANSACTION. NORMALIZED TO HAVE EUCLIDEAN NORM:")
	print(dict(zip(vocab, weighted_transaction_features)), "\n")
	
	print("WEIGHTS PER TOKEN:")
	print(dict(zip(vocab, numpy.around(transformer.idf_, decimals=5).tolist())), "\n")

def compareTokens(dataset_a, dataset_b):
	"""Compares available tokens"""

	deltas = {}
	print(type(dataset_a))

	for key in dataset_a:
		a_frequency = dataset_a[key]
		b_frequency = dataset_b.get(key, 0)
		deltas[key] = a_frequency - b_frequency

	print(OrderedDict(sorted(deltas.items(), key=lambda t: t[1])))

	#print(deltas)

if __name__ == "__main__":

	# Load Files from Datasets
	factual_merchants = load_dict_list("/home/ec2-user/1.5_Million_Factual.txt", delimiter="\t")
	factual_merchants = [string_cleanse(merchant["address"] + " " + merchant["name"] + " " + merchant["region"] + " " + merchant["locality"]) for merchant in factual_merchants]
	transactions = load_dict_list("/media/ephemeral0/unmasked_panels/writable_place/20140410_YODLEE_CARD_PANEL_UNMASKED_physical.txt")
	transactions = [string_cleanse(transaction["DESCRIPTION"]) for transaction in transactions]
	
	# TODO Strip numbers

	# Run Analysis
	factual_tokens = vectorize(factual_merchants)
	yodlee_tokens = vectorize(transactions)

	compareTokens(yodlee_tokens, factual_tokens)
#!/usr/local/bin/python3

"""This script collects the most common tokens in our transactions that 
are either not found in factual data, or found significantly less often"""

import random, numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def random_sample(transactions=None):
	"""Takes a bag of words approach to vectorizing our text corpus"""

	sample_size = 527
	randIndex = random.sample(range(len(transactions)), sample_size)
	content = [transactions[i] for i in randIndex]

	return content

def vectorize(transactions=None):
	"""Takes a bag of words approach to vectorizing our text corpus"""
	vectorizer = CountVectorizer(min_df=0.04, max_df=0.95)
	countVector = vectorizer.fit_transform(transactions).toarray()
	num_samples, num_features = countVector.shape
	vocab = vectorizer.get_feature_names()

	termWeighting(vocab, countVector, transactions)
	tokenCount(vocab, countVector, num_samples)

	return distribution_dict

def tokenCount(vocab, countVector, num_samples):

	numpy.clip(countVector, 0, 1, out=countVector)
	dist = numpy.sum(countVector, axis=0)
	dist = dist.tolist()

	print("TOKEN COUNT:")
	print(dict(zip(vocab, dist)), "\n")

	tokenFrequency(vocab, dist, num_samples)

def tokenFrequency(vocab, dist, num_samples):

	dist[:] = [x / num_samples for x in dist]
	dist = numpy.around(dist, decimals=2).tolist()
	distribution_dict = dict(zip(vocab, dist))

	print("TOKEN FREQUENCY:")
	print(distribution_dict)
	sys.exit()

def termWeighting(vocab, countVector, transactions):
	"""Takes a bag of words approach to vectorizing our text corpus"""
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(countVector)
	weighted_transaction_features = tfidf.toarray().tolist()[15]

	print(transactions[15], "\n")
	print("TOKEN IMPORTANCE SINGLE TRANSACTION. NORMALIZED TO HAVE EUCLIDEAN NORM:")
	print(dict(zip(vocab, weighted_transaction_features)), "\n")
	
	print("WEIGHTS PER TOKEN:")
	print(dict(zip(vocab, numpy.around(transformer.idf_, decimals=2).tolist())), "\n")	
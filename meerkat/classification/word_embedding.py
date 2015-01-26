#!/usr/local/bin/python3.3

"""This module generates semantic vector representations
from a given corpus of documents

Created on Nov 26, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 meerkat.classification.word_embedding [file_name]

#####################################################

import csv
import sys
import operator

import numpy
import pandas as pd
from pprint import pprint
from gensim.models import Word2Vec
from gensim.utils import tokenize
from sklearn.feature_extraction.text import CountVectorizer

from meerkat.various_tools import clean_bad_escapes

class documentGenerator(object):
	"""A memory efficient document loader"""

	def __init__(self, file_name):
		df = pd.read_csv(file_name, na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='\t', error_bad_lines=False)
		self.docs = df[['ENTITY', 'ENTITY_TYPE']]

	def __iter__(self):
		for doc in self.docs.iterrows():
			doc = ' '.join(doc[1]).lower()
			yield tokenize(doc)

def vectorize(corpus, min_df=1):
	"""Vectorize text corpus"""

	vectorizer = CountVectorizer(min_df=min_df, ngram_range=(1,1), stop_words='english')
	countVector = vectorizer.fit_transform(corpus).toarray()
	num_samples, num_features = countVector.shape
	vocab = vectorizer.get_feature_names()
	word_count = wordCount(vocab, countVector)
	sorted_words = sorted(word_count.items(), key=operator.itemgetter(1))

	pprint(sorted_words)

def wordCount(vocab, countVector):
	"""Count words"""

	numpy.clip(countVector, 0, 1, out=countVector)
	dist = numpy.sum(countVector, axis=0)
	dist = dist.tolist()
	word_count = dict(zip(vocab, dist))

	return word_count

def run_from_command_line(command_line_arguments):
	"""Runs the module when invoked from the command line."""

	#df = pd.read_csv(sys.argv[1], na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
	#corpus = df['DESCRIPTION_UNMASKED'].tolist()
	#vectorize(corpus, min_df=0.005)

	#cleaned = clean_bad_escapes(sys.argv[1])
	documents = documentGenerator(sys.argv[1])
	model = Word2Vec(documents)
	model.save('./models/w2v_factual')

if __name__ == "__main__":
	run_from_command_line(sys.argv)
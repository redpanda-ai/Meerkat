#!/usr/local/bin/python3.3
"""
A trie used to classify merchants in given transaction strings

Created on June 30, 2015
@author: Sivan Mehta
"""

import json
import csv
import sys
import string
import os
import marisa_trie as mt
import pickle
from pprint import pprint

def standardize(text):
	"""converts text to all caps, no punctuation, and no whitespace"""
	try:
		text = text.upper()
	except:
		return ""
	if len(text) == 0:
		return ""
	for space in string.whitespace:
		text = text.replace(space, "")
	for mark in string.punctuation:
		text = text.replace(mark, "")
	return text

def get_json_from_file(input_filename):
	"""Opens a file of JSON and returns a dictionary object"""
	try:
		input_file = open(input_filename, encoding='utf-8')
		my_json = json.loads(input_file.read())
		input_file.close()
		return my_json
	except IOError:
		print("{0} not found, aborting.".format(input_filename))
		sys.exit()
	return None

def create_merchant_trie(input_filename, output_filename):
	"""creates a trie from an input file and writes the trie to an output file"""
	merchant_json = get_json_from_file(input_filename)
	merchants = set()
	for key in merchant_json.keys():
		merchants.add(standardize(key))

	merchants = mt.Trie(merchants)

	merchants.save(output_filename)

def generate_merchant_trie():
	"""either loads the trie from a file or creates one if no file is found"""
	trie = mt.Trie()
	if os.path.isfile("meerkat/classification/models/merchant_trie.marisa"):
		trie.load("meerkat/classification/models/merchant_trie.marisa")
	else:
		trie = create_merchant_trie("meerkat/classification/label_maps/permanent_bank_label_map.json", \
			'meerkat/classification/models/merchant_trie.marisa')
	return trie

if __name__ == "__main__":
	create_merchant_trie("meerkat/classification/label_maps/top_1000_factual.json", 'meerkat/classification/models/merchant_trie.marisa')
	my_merchant_trie = generate_merchant_trie()
	# quick test to see if everything loaded
	# for key in json.loads(open("meerkat/classification/label_maps/top_1000_factual.json").read()):
	# 	print(standardize(key) in my_merchant_trie)
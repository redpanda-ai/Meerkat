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

	merchants.save('meerkat/classification/models/merchant_trie.marisa')

def create_bigger_merchant_trie():
	with open("data/input/us_places_factual.csv") as f:
		reader  = csv.reader(f, delimiter = "\t")

		merchants = set()

		stop = 100000
		seen = 0

		for line in reader:
			merchants.add(standardize(line[1]))
			# print("added ", standardize(line[1]))
			if seen > stop:
				break
			seen += 1
			sys.stdout.flush()
			sys.stdout.write("\rseen %d merchants of ~19 million" % seen)

		merchants = mt.Trie(merchants)

		merchants.save("meerkat/classification/models/us_factual_trie.marisa")

	return merchants

def generate_merchant_trie():
	"""either loads the trie from a file or creates one if no file is found"""
	trie = mt.Trie()
	if os.path.isfile("meerkat/classification/models/merchant_trie.marisa"):
		trie.load("meerkat/classification/models/merchant_trie.marisa")
	else:
		trie = create_merchant_trie("meerkat/classification/label_maps/permanent_bank_label_map.json", \
			'meerkat/classification/models/merchant_trie.marisa')
	return trie

def generate_bigger_merchant_trie():
	"""
		either loads the trie from a file or creates one if no file is found
	"""
	trie = mt.Trie()
	if os.path.isfile("meerkat/classification/models/us_factual_trie.marisa"):
		trie.load("meerkat/classification/models/us_factual_trie.marisa")
	else:
		trie = create_bigger_merchant_trie()
	return trie

if __name__ == "__main__":
	my_merchant_trie = generate_merchant_trie()
	new_merchant_trie = create_bigger_merchant_trie()
	# quick test to see if everything loaded
	# for key in json.loads(open("meerkat/classification/label_maps/permanent_bank_label_map.json").read()):
	# 	print(standardize(key) in my_merchant_trie)
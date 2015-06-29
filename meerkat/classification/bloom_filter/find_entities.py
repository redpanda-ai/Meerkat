#!/usr/local/bin/python3.3
#pylint: disable=mixed-indentation
"""Finds entities

Created on Dec 20, 2014
@author: J. Andrew Key

Modified on June 25, 2015
@author: Sivan Mehta
"""

import pandas as pd
import csv
import pickle
import os
import string

from .bloom import *

STATES = [
	"AL", "AK", "AZ", "AR", "CA", \
	"CO", "CT", "DE", "FL", "GA", \
	"HI", "ID", "IL", "IN", "IA", \
	"KS", "KY", "LA", "ME", "MD", \
	"MA", "MI", "MN", "MS", "MO", \
	"MT", "NE", "NV", "NH", "NJ", \
	"NM", "NY", "NC", "ND", "OH", \
	"OK", "OR", "PA", "RI", "SC", \
	"SD", "TN", "TX", "UT", "VT", \
	"VA", "WA", "WV", "WI", "WY" ]

LOCATION_BLOOM = get_location_bloom()

def generate_city_map():
	"""
		generates a dictionary with the following structure

		(city, state) : (zip, latitude, longitude)

		eg:
		(Beverly Hills, CA) : (90210, 34.0731, 118.3994)

	"""
	print("generate location map")

	csv_file = csv.reader(open("meerkat/classification/bloom_filter/assets/us_cities_small.csv"), \
		delimiter="\t")
	data = {}
	for row in csv_file:
		try:
			int(row[2]) # some of the rows don't acually record a state name
		except ValueError:
			data[(row[2].upper(), row[1])] = (row[0], row[3], row[4])

	pickle.dump(data, open("meerkat/classification/bloom_filter/assets/CITY_INFO.log", 'wb'))

	return data

CITY_INFO = {}

if os.path.isfile("meerkat/classification/bloom_filter/assets/CITY_INFO.log"):
	with open("meerkat/classification/bloom_filter/assets/CITY_INFO.log", 'rb') as fp:
	    CITY_INFO = pickle.load(fp)
else:
	CITY_INFO = generate_city_map()

def in_merchant_bloom(splits):
	"""checks whether or not the splits are in the merchant bloom filter"""
	if splits in my_merchant_bloom:
		return splits
	return None

def in_location_bloom(text):
	"""
		checks whether or not the text are in the location bloom filter,
		and returns all the geographic information that we know about that
		location

		looks for a two letter state, then iterates backwards until it
		finds a known string. for example:

		el paso tx ==> FOUND STATE
				^^
		el paso tx ==> Not a known town
		   ^^^^^^^

		el paso tx ==> Known town, return all known information
		^^^^^^^^^^
	"""
	if len(text) == 1:
		return False
	else:
		region = text[-2:]
		biggest = None
		for i in range(len(text) -1, -1, -1):
			locality = text[i:- 2]
			if (locality, region) in LOCATION_BLOOM:
				biggest = (locality, region)

		return biggest

def location_split(my_text):
	# Capitalize
	my_text = standardize(my_text)

	for i in range(len(my_text) - 1):
		if my_text[i:i+2] in STATES:
			place = in_location_bloom(my_text[:i+2])
			if place:
				return place
	return None

def merchant_split(my_text, **kwargs):
	splits = [x.upper() for x in my_text.split()]
	for i in range(len(splits)):
		_, name = splits[i], splits[i]
		count = 1
		while name in pm_bloom and (i + count) < len(splits):
			name = " ".join(splits[i:i+count])
			count += 1
	return None

##THINK!
#Red Roof Inn, a three term bloom filter.
# We should know if Red leads to Red Roof which in term leads to Red Roof Inn.
# If each merchant stored its partials, the logic should be:
# 1. Store each partial, including the entire term in partials.
# 2. When scanning, start at first term and scan partials,
#    find the longest partial you can and then check full merchant bloom
# 3. Note that you may wish to remove tokens that have
#    already been found as locations (could be important)

def main():
	"""runs the file"""
	print("find_entities")
	input_file = "meerkat/classification/bloom_filter/input_file.txt.gz"
	data_frame = pd.io.parsers.read_csv(input_file, sep="|", compression="gzip")
	descriptions = data_frame['DESCRIPTION_UNMASKED']
	location_bloom_results = descriptions.apply(location_split)
	#TODO: add merchant results
	#merchant_bloom_results = descriptions.apply(merchant_split, axis=0)

	combined = pd.concat([location_bloom_results, descriptions], axis=1)
	combined.columns = ['LOCATION', 'DESCRIPTION_UNMASKED']
	pd.set_option('max_rows', 10000)
	combined.to_csv("meerkat/classification/bloom_filter/entities.csv", \
		mode="w", sep="|", encoding="utf-8")
	print(combined)
	print(location_bloom_results.describe())

if __name__ == "__main__":
	# my_merchant_bloom, pm_bloom = get_merchant_bloom()
	main()

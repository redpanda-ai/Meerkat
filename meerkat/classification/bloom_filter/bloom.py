#!/usr/local/bin/python3.3
#pylint: disable=mixed-indentation, bad-continuation

"""Bloom filter stuff.

Created on Dec 19, 2014
@author: J. Andrew Key

Updated on June 29, 2015
@author: Sivan Mehta
"""

import json, sys, os, string, csv
from pybloom import ScalableBloomFilter

def standardize(text):
	"""converts text to all caps, no punctuation, and no whitespace"""
	text = text.upper()
	for space in string.whitespace:
		text = text.replace(space, "")
	for mark in string.punctuation:
		text = text.replace(mark, "")
	return text

def get_json_from_file(input_filename):
	"""Opens a file of JSON and returns a json object"""
	try:
		input_file = open(input_filename, encoding='utf-8')
		my_json = json.loads(input_file.read())
		input_file.close()
		return my_json
	except IOError:
		print("{0} not found, aborting.".format(input_filename))
		sys.exit()
	return None

def create_location_bloom(src_filename, dst_filename):
	"""Creates a bloom filter from the provided input file."""
	sbf = ScalableBloomFilter(initial_capacity=100000, error_rate=0.001,\
		mode=ScalableBloomFilter.SMALL_SET_GROWTH)
	locations = get_json_from_file(src_filename)

	states = locations["aggregations"]["states"]["buckets"]
	for state in states:
		state_name = state["key"].upper()
		for city in state["localities"]["buckets"]:
			city_name = standardize(city["key"])
			location = (city_name, state_name)
			sbf.add(location)

	'''
	try:
		open('meerkat/classification/bloom_filter/assets/csv_not_json')
	except:
		get_diff_json_csv()

	with open('meerkat/classification/bloom_filter/assets/csv_not_json') as f:
		for line in f:
			city, state = line.strip().split('\t')
			city_name = standardize(city)
			state_name = state.upper()
			location = (city_name, state_name)
			sbf.add(location)
	'''

	with open(dst_filename, "bw+") as location_bloom:
		sbf.tofile(location_bloom)

	return sbf

'''
def get_diff_json_csv():
	dict_json = dict()
	locations = get_json_from_file('meerkat/classification/bloom_filter/assets/locations.json')
	states = locations["aggregations"]["states"]["buckets"]
	for state in states:
		state_name = state["key"].upper()
		for city in state["localities"]["buckets"]:
			city_name = standardize(city["key"])
			location = (city_name, state_name)
			dict_json[location] = (city["key"], state_name)

	dict_csv = dict()
	csv_file = csv.reader(open("meerkat/classification/bloom_filter/assets/us_cities_larger.csv", encoding="utf-8"), delimiter="\t")
	for row in csv_file:
		try:
			int(row[2])
		except ValueError:
			city = row[2]
			state = row[4]
			location = (standardize(city), state)
			dict_csv[location] = (city, state)

	with open('meerkat/classification/bloom_filter/assets/csv_not_json', 'w') as f:
		for key in dict_csv.keys():
			if key not in dict_json:
				tup = dict_csv[key]
				f.write('{0}	{1}'.format(tup[0], tup[1]) + '\n')

	with open('meerkat/classification/bloom_filter/assets/json_not_csv', 'w') as f:
		for key in dict_json.keys():
			if key not in dict_csv:
				tup = dict_json[key]
				f.write('{0}	{1}'.format(tup[0], tup[1]) + '\n')
'''

def get_location_bloom():
	"""Attempts to fetch a bloom filter from a file, making a new bloom filter
	if that is not possible."""
	sbf = None
	try:
		sbf = ScalableBloomFilter.fromfile(open("meerkat/classification/bloom_filter/assets/location_bloom", "br"))
		print("Location bloom filter loaded from file.")
	except:
		print("Creating new bloom filter")
		sbf = create_location_bloom("meerkat/classification/bloom_filter/assets/locations.json", "meerkat/classification/bloom_filter/assets/location_bloom")
	return sbf

def test_bloom_filter(sbf):
	"""Does a basic run through US and Canadian locations, looking for probable
	membership."""
	canadian_locations = [
		("TORONTO", "ON"),
		("MONTREAL", "QB"),
		("CALGARY", "AB"),
		("OTTOWA", "ON"),
		("EDMONTON", "AB")
	]
	us_locations = [
		("BOISE", "ID"),
		("HIGHGATE FALLS", "VT"),
		("SAN JOSE", "CA"),
		("WEST MONROE", "LA"),
		("DILLARD", "GA"),
		("FAKE CITY", "CA"),
		("CARSON CITY", "NV"),
		("SAINT LOUIS", "MO"),
		("SUNNYVALE", "CA")
	]
	print("Touring Canada")
	line = "{0} in bloom filter: {1}"
	for location in canadian_locations:
		print(line.format(location, location in sbf))
	print("Touring United States")
	for location in us_locations:
		print(line.format(location, location in sbf))

# MAIN PROGRAM
if __name__ == "__main__":
	my_location_bloom = get_location_bloom()
	

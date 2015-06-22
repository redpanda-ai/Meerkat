#!/usr/local/bin/python3.3
#pylint: disable=mixed-indentation, bad-continuation

"""Bloom filter stuff.

Created on Dec 19, 2014
@author: J. Andrew Key
"""

import json, sys, os
from pybloom import ScalableBloomFilter

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
			city_name = city["key"].upper()
			location = (city_name, state_name)
			sbf.add(location)

	with open(dst_filename, "bw+") as location_bloom:
		sbf.tofile(location_bloom)

	return sbf

def create_merchant_bloom(src_filename, dst_filename, partial_filename):
	"""Creates a bloom filter from the provided input file."""
	m_bloom = ScalableBloomFilter(initial_capacity=5000, error_rate=0.001,\
		mode=ScalableBloomFilter.SMALL_SET_GROWTH)
	p_bloom = ScalableBloomFilter(initial_capacity=5000, error_rate=0.001,\
		mode=ScalableBloomFilter.SMALL_SET_GROWTH)
	merchants = get_json_from_file(src_filename)

	count = 0
	buckets = merchants["aggregations"]["merchants"]["buckets"]
	#Iterate through merchant names
	for bucket in buckets:
		if count >= 500: # limit
			break
		count += 1
		merchant_name = bucket["key"].upper()
		#Add full merchant names to merchant bloom
		print("Adding '{0}'".format(merchant_name))
		m_bloom.add(merchant_name)
		#Handle partial merchant names to partial bloom
		splits = merchant_name.split()
		if len(splits) > 1:
			for i in range(1, len(splits)):
				partial = " ".join(splits[:i])
				print("\tAdding '{0}'".format(partial))
				p_bloom.add(partial)

	with open(dst_filename, "bw+") as merchant_bloom:
		m_bloom.tofile(merchant_bloom)
	with open(partial_filename, "bw+") as partial_bloom:
		p_bloom.tofile(partial_bloom)

	return m_bloom, p_bloom


def get_merchant_bloom():
	"""Attempts to fetch a bloom filter from a file, making a new bloom filter
	if that is not possible."""
	sbf, partial = None, None
	# check if files exist
	if os.path.isfile("stats/merchant_bloom") and \
	   os.path.isfile("stats/partial_merchant_bloom"):
		print("Creating merchant bloom filters")
		sbf, partial = create_merchant_bloom("stats/merchant_names.json", "stats/merchant_bloom", "stats/partial_merchant_bloom")
	else:
		sbf = ScalableBloomFilter.fromfile(open("stats/merchant_bloom", "br"))
		partial = ScalableBloomFilter.fromfile(open("stats/partial_merchant_bloom", "br"))
		print("Full merchant bloom filter loaded from file.")
		print("Partial merchant bloom filter loaded from file.")
	return sbf, partial

def get_location_bloom():
	"""Attempts to fetch a bloom filter from a file, making a new bloom filter
	if that is not possible."""
	sbf = None
	try:
		sbf = ScalableBloomFilter.fromfile(open("meerkat/bloom_filter/assets/location_bloom", "br"))
		print("Location bloom filter loaded from file.")
	except:
		print("Creating new bloom filter")
		sbf = create_location_bloom("meerkat/bloom_filter/assets/locations.json", "meerkat/bloom_filter/assets/location_bloom")
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
	#test_bloom_filter(my_location_bloom)
	# my_merchant_bloom, my_partial_merchant_bloom = get_merchant_bloom()

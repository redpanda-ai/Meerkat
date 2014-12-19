#!/bin/python3.3
"""Bloom filter stuff."""
import json
import sys
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

def create_bloom_filter(src_filename, dst_filename):
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

def get_bloom_filter():
	sbf = None
	try:
		sbf = ScalableBloomFilter.fromfile(open("stats/location_bloom", "br"))
		print("Bloom filter loaded from file.")
	except:
		print("Creating new bloom filter")
		sbf = create_bloom_filter("stats/locations.json", "stats/location_bloom")
	return sbf

def test_bloom_filter(sbf):
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
		("DILLARD", "GA")
	]
	print("Touring Canada")
	line = "{0} in bloom filter: {1}"
	for location in canadian_locations:
		print(line.format(location, location in sbf))
	print("Touring United States")
	for location in us_locations:
		print(line.format(location, location in sbf))

my_bloom_filter = get_bloom_filter()
test_bloom_filter(my_bloom_filter)

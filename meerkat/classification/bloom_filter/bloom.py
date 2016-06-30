#!/usr/local/bin/python3.3
#pylint: disable=mixed-indentation, bad-continuation

"""Bloom filter stuff.

Created on Dec 19, 2014
@author: J. Andrew Key

Updated on June 29, 2015
@author: Sivan Mehta
"""

import json, sys, csv
from pybloom import ScalableBloomFilter

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
	"VA", "WA", "WV", "WI", "WY", \
	"DC"]

SUBS = {
	# Directions
	"EAST": "E",
	"WEST": "W",
	"NORTH": "N",
	"SOUTH": "S",

	# Abbreviations
	"SAINT": "ST",
	"FORT": "FT",
	"CITY" : ""

	# can get more in the future
}

FORMAT = {
	"E": "EAST",
	"W": "WEST",
	"N": "NORTH",
	"S": "SOUTH",

	"ST": "SAINT",
	"FT": "FORT"
}


def standardize(text):
	"""converts text to all caps, no punctuation, and no whitespace"""
	text = text.upper()
	text = ''.join(text.split())
	for mark in '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~':
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

def get_city_subs(city):
	city = city.upper()
	city = city.replace('.', '')
	parts = city.split()
	length = len(parts)
	def dfs(step, path):
		if step == length:
			subs.append(path)
			return
		if parts[step] not in SUBS:
			dfs(step + 1, path + [parts[step]])
		else:
			dfs(step + 1, path + [parts[step]])
			dfs(step + 1, path + [SUBS[parts[step]]])
	subs = []
	dfs(0, [])
	subs = [''.join(sub) for sub in subs]
	return subs

def format_city_name(city):
	city = city.upper()
	city = city.replace('.', '')
	parts = city.split()
	for i in range(len(parts)):
		if parts[i] in FORMAT: parts[i] = FORMAT[parts[i]]
		parts[i] = parts[i][0] + parts[i][1:].lower() if len(parts[i]) > 1 else parts[i]
	return ' '.join(parts)

def create_location_bloom(csv_filename, json_filename, dst_filename):
	"""Creates a bloom filter from the provided input file."""
	sbf = ScalableBloomFilter(initial_capacity=100000, error_rate=0.001,\
		mode=ScalableBloomFilter.SMALL_SET_GROWTH)

	# add us_cities_larger.csv file
	csv_file = csv.reader(open(csv_filename, encoding="utf-8"), delimiter="\t")
	for row in csv_file:
		try:
			int(row[2]) # some of the rows don't acually record a state name
		except ValueError:
			city = row[2]
			state = row[4].upper()
			if state in STATES:
				subs = get_city_subs(city)
				for sub in subs:
					location = (standardize(sub), state)
					sbf.add(location)

	# add locations.json file
	locations = get_json_from_file(json_filename)
	states = locations["aggregations"]["states"]["buckets"]
	for state in states:
		state_name = state["key"].upper()
		for city in state["localities"]["buckets"]:
			city_name = format_city_name(city["key"])
			subs = get_city_subs(city_name)
			for sub in subs:
				location = (standardize(sub), state_name)
				sbf.add(location)

	with open(dst_filename, "bw+") as location_bloom:
		sbf.tofile(location_bloom)

	return sbf

def get_location_bloom(filename):
	"""Attempts to fetch a bloom filter from a file, making a new bloom filter
	if that is not possible."""
	sbf = None
	try:
		sbf = ScalableBloomFilter.fromfile(\
			open(filename, "br"))
		print("Location bloom filter loaded from file.")
	except:
		print("Creating new bloom filter")
		sbf = create_location_bloom('meerkat/classification/bloom_filter/assets/us_cities_larger.csv', \
			"meerkat/classification/bloom_filter/assets/locations.json", filename)
	return sbf

# MAIN PROGRAM
if __name__ == "__main__":
	get_location_bloom()

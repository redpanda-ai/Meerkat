#!/usr/local/bin/python3.3
#pylint: disable=mixed-indentation
"""Finds entities

Created on Dec 20, 2014
@author: J. Andrew Key

Modified on June 29, 2015
@author: Sivan Mehta
"""

import csv
import pickle
import os

import pandas as pd
from .bloom import *
from .generate_json import generate_js

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

LOCATION_BLOOM = get_location_bloom()

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

def add_with_subs(data, city, state):
	proper = "%s%s" % (city, state)
	standard = standardize(proper)
	proper = (city, state)
	data[standard] = proper
	for sub in SUBS.keys():
		if sub in standard:
			standard = standard.replace(sub, SUBS[sub])
			data[standard] = proper

def generate_city_map():
	"""
		generates a dictionary with the following structure

		(CityState) : (City, State)

		eg:
		(BEVERLYHILLSCA) : (Beverly Hills, CA)

	"""
	print("generate location map")

	data = {}

	try:
		open("meerkat/classification/bloom_filter/assets/json_not_csv")
	except:
		get_diff_json_csv()

	with open("meerkat/classification/bloom_filter/assets/json_not_csv") as f:
		for line in f:
			city, state = line.strip().split('\t')
			add_with_subs(data, city, state)

	csv_file = csv.reader(open("meerkat/classification/bloom_filter/assets/us_cities_larger.csv", \
		encoding="utf-8"), delimiter="\t")

	for row in csv_file:
		try:
			int(row[2]) # some of the rows don't acually record a state name
		except ValueError:
			city = row[2]
			state = row[4] # for larger.csv
			# state = row[1] # for small.csv
			add_with_subs(data, city, state)

	pickle.dump(data, open("meerkat/classification/bloom_filter/assets/CITY_SUBS", 'wb'))

	return data

CITY_SUBS = {}

if os.path.isfile("meerkat/classification/bloom_filter/assets/CITY_SUBS"):
	with open("meerkat/classification/bloom_filter/assets/CITY_SUBS", 'rb') as fp:
	    CITY_SUBS = pickle.load(fp)
else:
	CITY_SUBS = generate_city_map()

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
	if len(text) <= 2:
		return False
	else:
		region = text[-2:]
		biggest = None

		for i in range(len(text) - 3, -1, -1):
			locality = text[i:-2]
			if (locality, region) in LOCATION_BLOOM:
				biggest = (locality, region)

		return biggest

def location_split(my_text):
	# Capitalize and remove spaces
	tag = tag_text(my_text)
	my_text = standardize(my_text)

	try:
		open('meerkat/classification/bloom_filter/assets/words_start_with_states.json')
	except:
		generate_js()

	words = get_json_from_file('meerkat/classification/bloom_filter/assets/words_start_with_states.json')
	length = len(my_text)
	for i in range(length - 2, -1, -1):
		if my_text[i:i+2] in STATES and tag[i+1] == 'C' and get_word(tag, my_text, i) not in words[my_text[i:i+2]]:
			place = in_location_bloom(my_text[:i+2])
			if place:
				key = place[0] + place[1]
				try: return CITY_SUBS[key]
				except: pass
	return None

def get_word(tag, text, index):
	'''get the word concact with state'''
	end = index + 2
	for ch in tag[index + 2:]:
		if ch == 'C':
			end += 1
		else:
			break
	return text[index:end]

def tag_text(text):
	'''make tag for text'''
	text = text.replace('\'', '')
	for mark in '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~':
		text = text.replace(mark, ' ')
	text = text.strip()
	tag = ''
	for i in range(len(text)):
		if text[i] == ' ':
			continue
		elif i == 0 or text[i - 1] == ' ':
			tag += 'B'
		else:
			tag += 'C'
	return tag

def main():
	"""runs the file"""
	print("find_entities")
	input_file = "meerkat/classification/bloom_filter/input_file.txt.gz"
	# input_file = "data/input/should_search_labels.csv.gz"
	data_frame = pd.io.parsers.read_csv(input_file, sep="|", compression="gzip")
	descriptions = data_frame['DESCRIPTION_UNMASKED']
	location_bloom_results = descriptions.apply(location_split)

	combined = pd.concat([location_bloom_results, descriptions], axis=1)
	combined.columns = ['LOCATION', 'DESCRIPTION_UNMASKED']

	pd.set_option('max_rows', 10000)
	combined.to_csv("meerkat/classification/bloom_filter/entities.csv", \
		mode="w", sep="|", encoding="utf-8")
	print(combined)
	print(location_bloom_results.describe())

if __name__ == "__main__":
#	main()
	print(location_split('VICTORIAS'))

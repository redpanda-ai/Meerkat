#!/usr/local/bin/python3.3

"""Finds entities

Created on Dec 20, 2014
@author: J. Andrew Key
"""

import pandas as pd

from .bloom import get_location_bloom, get_merchant_bloom

STATES = {
	"AL": "", "AK": "", "AZ": "", "AR": "", "CA": "",
	"CO": "", "CT": "", "DE": "", "FL": "", "GA": "",
	"HI": "", "ID": "", "IL": "", "IN": "", "IA": "",
	"KS": "", "KY": "", "LA": "", "ME": "", "MD": "",
	"MA": "", "MI": "", "MN": "", "MS": "", "MO": "",
	"MT": "", "NE": "", "NV": "", "NH": "", "NJ": "",
	"NM": "", "NY": "", "NC": "", "ND": "", "OH": "",
	"OK": "", "OR": "", "PA": "", "RI": "", "SC": "",
	"SD": "", "TN": "", "TX": "", "UT": "", "VT": "",
	"VA": "", "WA": "", "WV": "", "WI": "", "WY": ""
	}

def in_merchant_bloom(splits):
	"""checks whether or not splits are in my_merchant_bloom"""
	if splits in my_merchant_bloom:
		return splits
	return None

def in_location_bloom(splits):
	"""checks whether or not splits are in my_location"""
	len_splits = len(splits)
	if len_splits == 1:
		return None
	else:
		region, before = splits[len_splits-1], splits[:len_splits-1]
		for i in range(len(before)):
			locality = " ".join(before[i:])
			place = (locality, region)
			if place in my_bloom:
				return place
	return None

def location_split(my_text):
	"""returns first location in text that is in the location bloom"""
	splits = [x.upper() for x in my_text.split()]
	for i in range(len(splits)):
		if splits[i] in STATES:
			place = in_location_bloom(splits[:i+1])
			if place:
				return place
	return None

def merchant_split(my_text):
	splits = [x.upper() for x in my_text.split()]
	for i in range(len(splits)):
		previous, name = splits[i], splits[i]
		count = 1
		#print("START")
		while name in pm_bloom and (i + count) < len(splits):
			previous = name
			#print("\tFound {0}".format(name))
			name = " ".join(splits[i:i+count])
			#print("New name is {0}".format(name))
			count += 1
		#print("Searching for '{0}'".format(previous))
		if previous in my_merchant_bloom:
			#print("\tFound {0}".format(previous))
			return name
	return None

##THINK!
#Red Roof Inn, a three term bloom filter.
#We should know if Red leads to Red Roof which in term leads to Red Roof Inn.
#If each merchant stored its partials, the logic should be:
#1. Store each partial, including the entire term in partials.
#2. When scanning, start at first term and scan partials, find the 
#	longest partial you can and then check full merchant bloom
#3. Note that you may wish to remove tokens that have already been 
#	found as locations (could be important)

def main():
	"""Runs the file"""
	print("find_entities")
	input_file = "bank.txt.gz"
	data_frame = pd.io.parsers.read_csv(input_file, sep="|", compression="gzip")
	descriptions = data_frame['DESCRIPTION_UNMASKED']
	location_bloom_results = descriptions.apply(location_split)

	merchant_bloom_results = descriptions.apply(merchant_split)

	combined = pd.concat([location_bloom_results, merchant_bloom_results, descriptions], axis=1)
	combined.columns = ['LOCATION', 'MERCHANT_NAME', 'DESCRIPTION_UNMASKED']
	pd.set_option('max_rows', 10000)
	combined.to_csv("entities.csv", mode="w", sep="|", encoding="utf-8")

if __name__ == "__main__":
	my_bloom = get_location_bloom()
	my_merchant_bloom, pm_bloom = get_merchant_bloom()
	main()

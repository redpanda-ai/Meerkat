#!/usr/local/bin/python3.3

"""Finds entities

Created on Dec 20, 2014
@author: J. Andrew Key
"""

import nltk
import gzip
import pandas as pd
import sys
import re

from pybloom import ScalableBloomFilter
from .bloom import get_location_bloom, get_merchant_bloom

states = {
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
	if splits in my_merchant_bloom:
		return splits
	return None

def in_location_bloom(splits):
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

def location_split(my_text, **kwargs):
	splits = [x.upper() for x in my_text.split()]
	for i in range(len(splits)):
		if splits[i] in states:
			place = in_location_bloom(splits[:i+1])
			if place:
				return place
	return None

def merchant_split(my_text, **kwargs):
	splits = [x.upper() for x in my_text.split()]
	for i in range(len(splits)):
		previous, name = splits[i], splits[i]
		c = 1
		#print("START")
		while name in pm_bloom and (i + c) < len(splits):
			previous = name
			#print("\tFound {0}".format(name))
			name = " ".join(splits[i:i+c])
			#print("New name is {0}".format(name))
			c += 1
		#print("Searching for '{0}'".format(previous))
		if previous in my_merchant_bloom:
			#print("\tFound {0}".format(previous))
			return name
	return None
#	for i in range(len(splits)):
#		merchant = in_merchant_bloom(splits[i])
#		if merchant:
#			return merchant
#	return None

##THINK!
#Red Roof Inn, a three term bloom filter.
#We should know if Red leads to Red Roof which in term leads to Red Roof Inn.
#If each merchant stored its partials, the logic should be:
#1. Store each partial, including the entire term in partials.
#2. When scanning, start at first term and scan partials, find the longest partial you can and then check full merchant bloom
#3. Note that you may wish to remove tokens that have already been found as locations (could be important)

def start():
	print("find_entities")
	input_file = "bank.txt.gz"
	data_frame = pd.io.parsers.read_csv(input_file,sep="|",compression="gzip")
	descriptions = data_frame['DESCRIPTION_UNMASKED']
	location_bloom_results = descriptions.apply(location_split, axis=0)
	#TODO: add merchant results
	merchant_bloom_results = descriptions.apply(merchant_split, axis=0)

	#combined = pd.concat([location_bloom_results, descriptions], axis=1)
	#combined.columns = ['NER - LOCATION', 'DESCRIPTION_UNMASKED']
	combined = pd.concat([location_bloom_results, merchant_bloom_results, descriptions], axis=1)
	combined.columns = ['LOCATION', 'MERCHANT_NAME', 'DESCRIPTION_UNMASKED']
	pd.set_option('max_rows', 10000)
	#pd.set_option('display.max_colwidth', 60)
	#chunk.to_csv(dst_local_path + dst_file_name, columns=header, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)
	combined.to_csv("entities.csv", mode="w", sep="|", encoding="utf-8")
	#print(combined)
	#print(location_bloom_results.describe())
if __name__ == "__main__":
	my_bloom = get_location_bloom()
	my_merchant_bloom, pm_bloom = get_merchant_bloom()
	start()

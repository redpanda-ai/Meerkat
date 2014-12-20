import nltk
import gzip
import pandas as pd
import sys
import re

from pybloom import ScalableBloomFilter
from .bloom import get_bloom_filter

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

def in_bloom(splits):
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

def my_split(my_text, **kwargs):
	splits = [x.upper() for x in my_text.split()]
	for i in range(len(splits)):
		if splits[i] in states:
			place = in_bloom(splits[:i+1])
			if place:
				return place
	return None

def start():
	print("find_entities")
	input_file = "bank.txt.gz"
	data_frame = pd.io.parsers.read_csv(input_file,sep="|",compression="gzip")
	descriptions = data_frame['DESCRIPTION_UNMASKED']
	bloom_results = descriptions.apply(my_split, axis=0)
	combined = pd.concat([bloom_results, descriptions], axis=1)
	combined.columns = ['NER - LOCATION', 'DESCRIPTION_UNMASKED']
	pd.set_option('max_rows', 5000)
	#pd.set_option('display.max_colwidth', 60)
	print(combined)
if __name__ == "__main__":
	my_bloom = get_bloom_filter()
	start()

"""
Created July 13, 2015
@author: Sivan Mehta
"""

from __future__ import print_function
from pprint import pprint
import operator
import pandas as pd
import numpy as np
import sys
import time

from meerkat.classification.merchant_trie import standardize

"""
	This module generates a list of the top 1000 merchants in the factual data
"""

def main():
	"""
		generates a list of the top 1000 most popular merchants in 
		data/input/us_places_factual.log (which is really a csv)

		you can pull the file with the following commands:
		$ s3cmd get s3://yodleefactual/us_places.factual.2014_05_01.1398986783000.tab.gz

		$ gunzip us_places.factual.2014_05_01.1398986783000.tab.gz data/input/us_places_factual.log

		The most recent runtime for this process is 2:37.17
	"""

	reader = pd.read_csv("data/input/us_places_factual.log", chunksize=10000, na_filter=False, encoding="utf-8", sep='\t', error_bad_lines=False)
	counts = {}

	count = 1
	start = time.time()

	for df in reader:
		df['name'] = df['name'].astype('category')
		value_counts = df['name'].value_counts()
		
		for item, value in value_counts.iteritems():
			item = standardize(item)
			if item in counts:
				counts[item] += value
			else:
				counts[item] = value

		if count <= 2059: # number of chunks in the input file
			sys.stdout.flush()
			sys.stdout.write("\r%d of 2058 chunks seen after %2.2fs... " % (count, time.time() - start))
			count += 1
		else:
			break

	print("done!")

	print("trimming keys... ", end = "")
	trimmed = {}
	for key in counts.keys():
		if counts[key] >= 100:
			trimmed[key] = counts[key]

	counts = trimmed
	print("done!")


	sorted_counts = sorted(counts.items(), key=operator.itemgetter(1))[-1000:]
	top_1000 = {}
	for key, value in sorted_counts:
		top_1000[key] = value

	f = open('meerkat/classification/label_maps/top_1000_factual.txt','w')
	pprint(top_1000, f)

if __name__ == "__main__":
	main()
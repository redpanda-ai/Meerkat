from __future__ import print_function
from pprint import pprint
import operator
import pandas as pd
import numpy as np
import sys, time

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
	"""

	reader = pd.read_csv("data/input/us_places_factual.log", chunksize=10000, na_filter=False, encoding="utf-8", sep='\t', error_bad_lines=False)
	counts = {}

	count = 1

	for df in reader:
		df['name'] = df['name'].astype('category')
		value_counts = df['name'].value_counts().head(15000)
		
		for item, value in value_counts.iteritems():
			if item in counts:
				counts[item] += value
			else:
				counts[item] = value

		if count <= 2059: # number of chunks in the input file
			sys.stdout.flush()
			sys.stdout.write("\r%d of 2058 chunks seen... " % count)
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

	f = open('meerkat/classification/label_maps/top_1000_factual.txt','w')

	sorted_counts = sorted(counts.items(), key=operator.itemgetter(1))
	print(sorted_counts[-1000:], file = f)
if __name__ == "__main__":
	main()
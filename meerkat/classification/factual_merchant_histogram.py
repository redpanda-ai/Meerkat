"""
	This module generates a list of the top 1000 merchants in the factual data
"""


import pandas as pd
import numpy as np

from meerkat.classification.merchant_trie import standardize

def main():
	"""
		generates a list of the top 1000 most popular merchants in 
		data/input/us_places_factual.log (which is really a csv)

		you can pull the file with the following command:
		s3cmd get s3://yodleefactual/us_places.factual.2014_05_01.1398986783000.tab.gz
	"""

	data = pd.read_csv("data/input/us_places_factual.log", sep="\t", chunksize=10)

	merchants = pd.Series()

	maxCount = 1000
	count = 0
	for chunk in data:
		for merchant in chunk.name:
			merchants.set_value(count, merchant)
			count += 1

		if count > maxCount:
			break

	count = 1
	top_merchants = merchants.value_counts()[:1000]

	print(top_merchants)

if __name__ == "__main__":
	main()
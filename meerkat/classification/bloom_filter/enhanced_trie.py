import sys
import pandas as pd
import csv
from meerkat.various_tools import load_params
from meerkat.classification.bloom_filter import bloomfilter_unique_city
from meerkat.classification.bloom_filter import trie

CITY_BLOOM = bloomfilter_unique_city.get_location_bloom()
MIN_LEN, MAX_LEN = bloomfilter_unique_city.city_length_range()

def location_split(my_text):
	trie_result = trie.location_split(my_text)
	if trie_result:
		return trie_result[0].upper(), trie_result[1].upper()

	json_filename = "meerkat/classification/bloom_filter/assets/Starbucks_unique_city_state.json"
	unique_city_state_dict = load_params(json_filename)

	transaction_word = ['PURCHASE']
	my_text = my_text.upper()
	for i in range(len(my_text) - 1, -1, -1):
		for j in range(MIN_LEN, MAX_LEN + 1):
			if i - j + 1 < 0:
				continue
			city = my_text[i - j + 1: i + 1]
			if (city in CITY_BLOOM and city in unique_city_state_dict and
				city not in transaction_word):
				return city, unique_city_state_dict[city]
	return None

def main():
	input_file = sys.argv[1]
	df = pd.read_csv(input_file, sep="\t")
	descriptions = df['question']
	location_bloom_results = descriptions.apply(location_split)

	combined = pd.concat([location_bloom_results, descriptions], axis=1)
	combined.columns = ['LOCATION', 'DESCRIPTION_UNMASKED']
	print(combined)
	#combined.to_csv("meerkat/classification/bloom_filter/unique_cities.csv", \
	#	mode="w", sep="\t", encoding="utf-8")

if __name__ == "__main__":
	main()

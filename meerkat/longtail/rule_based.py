#!/usr/local/bin/python3

"""Rule based merchant name extractor"""

############################################# USAGE ###############################################

#python3 -m meerkat.longtail_handler.rule_based

###################################################################################################

import re
import csv
import pandas as pd

from meerkat.various_tools import load_params

def get_city_state(city_dict):
	temp = []
	for state in city_dict:
		temp += [" ".join([city, state]).lower() for city in city_dict[state]]
	return temp

def bigram_search(str_list, subresults):
	bigrams = zip(str_list, str_list[1:])
	bigrams = [" ".join(item) for item in bigrams]
	temp = []
	for bigram in bigrams:
		for item in subresults:
			if re.search(r'\b' + bigram + r'\b', item):
				temp.append(item)
	if len(temp)>0:
		return temp
	return subresults

def find_merchant(string, city_list, merchants_path):
	results = []
	string = " ".join(string.lower().split())
	for city in city_list:
		if city in string:
			string = re.sub(city, "", string)
			break
	string = string.replace("*", " ")
	str_list = string.split()
	for word in str_list:
		results += check_acronym(word, merchants_path)
	results = unigram_search(str_list, results)
	return list(set(results))

def unigram_search(str_list, results):
	subresults = []
	for word in str_list:
		for item in results:
			if re.search(r'\b' + word + r'\b', item):
				subresults.append(item)
	if len(subresults) > 1:
		subresults = bigram_search(str_list, subresults)
	if len(subresults) > 0:
		return subresults
	return results

def check_acronym(word, merchants_path):
	candidates = []
	"""
	for merchant in merchants_list:
		candidate = merchant
	"""
	f = csv.reader(open(merchants_path), delimiter='\t')
	header = next(f)
	for merchant in f:
		merchant = merchant[1]
		candidate = merchant
		found = True
		for i in range(len(word)):
			char = word[i]
			try:
				index = candidate.index(char)
				if len(candidate) > 1:
					candidate = candidate[index+1:]
			except ValueError:
				found = False
				break
		if i == len(word) - 1:
			if word[i] in merchant.split()[-1] and word[0] in merchant.split()[0] and found:
				candidates.append(merchant)
	return candidates

def main():
	string = input("Enter description")
	# df = pd.read_csv("./meerkat/longtail_handler/String_based_merchant_identification_Final.csv", sep='|')
	# merchants_list = list(df["Tagged_merchant_string"].unique())
	# merchants_list = [" ".join(item.lower().split()) for item in merchants_list]
	# merchants_list = list(set(merchants_list))
	city_list = get_city_state(load_params("./meerkat/longtail_handler/state_city_zip.json"))
	results = find_merchant(string, city_list, "./meerkat/longtail_handler/factual.tab")
	print("Possible merchants are {0}.".format(results))

if __name__ == "__main__":
	main()

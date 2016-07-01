import sys
import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="csv containing city, state, and zip")
args = parser.parse_args()
my_dict = {}
with open(args.input_file) as infile:
	for line in infile:
		parts = line.split("\t")
		city = parts[2].upper()
		state = parts[4].upper()
		zipcode = parts[1]
		if state not in my_dict:
			my_dict[state] = {}
		if city not in my_dict[state]:
			my_dict[state][city] = [zipcode]
		else:
			my_dict[state][city].append(zipcode)

my_json = json.dumps(my_dict, sort_keys=True, indent=4, separators=(',', ': '))
print(my_json)


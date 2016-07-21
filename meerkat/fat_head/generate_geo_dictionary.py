import argparse
import csv
import json
import logging
import sys

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file", help="csv containing city, state, and zip")
	return parser.parse_args(args)

def get_geo_dict(input_file):
	my_dict = {}
	with open(input_file) as infile:
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
	return my_json

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	args = parse_arguments(sys.argv[1:])
	logging.info(get_geo_dict(args.input_file))


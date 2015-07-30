#!/usr/local/bin/python3.3

"""This module generates sample files
for human labelers to evaluate in aid
of training Meerkat.

Created on July 2, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: Transactions file must be pipe delimited.

# python3.3 make_pulses.py [file_to_split]
# python3.3 make_pulses.py card_sample_physical.txt

#####################################################

import csv
import sys
import collections
import random
from pprint import pprint

def load_dict_list(file_name, encoding='utf-8', delimiter="|"):
	"""Loads a dictionary of input from a file into a list."""
	input_file = open(file_name, encoding=encoding, errors='replace')
	dict_list = list(csv.DictReader(input_file, delimiter=delimiter,
		quoting=csv.QUOTE_NONE))
	input_file.close()
	return dict_list

def write_dict_list(dict_list, file_name):
	""" Saves a lists of dicts with uniform keys to file """

	with open(file_name, 'w') as output_file:
		dict_w = csv.DictWriter(output_file, \
								delimiter="|", \
								fieldnames=dict_list[0].keys(),\
								extrasaction='ignore')
		dict_w.writeheader()
		dict_w.writerows(dict_list)

def verify_arguments():
	"""Verifies proper usage"""

	sufficient_arguments = (len(sys.argv) == 2)

	if not sufficient_arguments:
		print("Insufficient arguments. Please see usage")
		sys.exit()

	dataset = sys.argv[1]
	dataset_included = dataset.endswith(".txt")

	if not dataset_included:
		print("Erroneous arguments. Please see usage")
		sys.exit()

def write_pulses(pulses):
	"""Take split pulses and write to file"""

	for i in range(len(pulses)):
		pulse = pulses[i]
		#TODO support both card and bank automatically
		pulse_prefix = "/mnt/ephemeral/training_data/pulses/bank/bank_sample_"
		file_name = pulse_prefix + str(i) + ".txt"
		write_dict_list(pulse, file_name)

def split_sample(filepath):
	"""Split sample into desired chunks"""

	data = load_dict_list(filepath)
	users = collections.defaultdict(list)

	# Split into user buckets
	for row in data:
		user = row.get("UNIQUE_MEM_ID", "")
		users[user].append(row)

	print("Number of Users: " + str(len(users.items())))

	# Limit # of Transactions per user
	for key, _ in users.items():
		user_data = users[key]
		if len(user_data) >= 25:
			users[key] = random.sample(user_data, 25)

	# Convert to List of Lists
	list_of_lists = []
	for key, _ in users.items():
		list_of_lists.append(users[key])

	# 15 users per pulse
	pulses = []
	while len(list_of_lists) > 0:
		pulse = []
		for _ in range(15):
			if len(list_of_lists) > 0:
				pulse = pulse + list_of_lists.pop()
		pulses.append(pulse)

	print("Number of Pulses: " + str(len(pulses)))

	return pulses

def run_from_command_line():
	"""Runs the module when invoked from the command line."""
	
	verify_arguments()
	pulses = split_sample(sys.argv[1])
	pprint(pulses);
	#write_pulses(pulses)

if __name__ == "__main__":
	run_from_command_line()

#!/bin/python3.3
"""This module is used to create random samples from panel files."""
import glob
import gzip
import logging
import os
import sys
from random import shuffle

def clean_line(line):
	"""Strips garbarge from both ends of a line, rendering it clean."""
	return str(line)[2:-3]

def get_header(input_file):
	"""Gets the header from an input file."""
	logging.critical("Getting header from %s", input_file)
	with gzip.open(input_file, "rb") as gzipped_input:
		#is_first_line = True
		for line in gzipped_input:
			return clean_line(line)

def count_me(input_file):
	logging.critical("Counting %s", input_file)
	with gzip.open(input_file, "rb") as gzipped_input:
		count = -1 #ignore the header
		for line in gzipped_input:
			count += 1
	return input_file, count

def produce_sample(sample_indices, file_index, sorted_filenames):
	"""Creates a dictionary showing which files contain the samples."""
	sample_indices.reverse()
	my_index = None
	result_dict = {}

	for file_name in sorted_filenames:
		lower, upper = file_index[file_name]
		in_bounds = True
		while sample_indices and in_bounds:
			my_index = sample_indices.pop()
			if my_index >= lower and my_index <= upper:
				if file_name not in result_dict:
					result_dict[file_name] = []
				result_dict[file_name].append(my_index)
			else:
				in_bounds = False
				sample_indices.append(my_index)
	return result_dict

def start(input_path):
	"""Runs the main program."""
	os.chdir(input_path)
	input_files = sorted(glob.glob('*.gz'))
	#ignore old sample
	input_files = [ x for x in input_files if x != "sample.txt.gz" ]
	#get the header
	header = get_header(input_files[0])
	#get a count for each input file
	counts = [count_me(x) for x in input_files]
	#obtain a total for all files
	total_list = [x[1] for x in counts]
	total = sum(total_list)

	#create an index of files to ranges
	ind = {}
	curr_count = 0
	for f_name, count in counts:
		temp = curr_count
		curr_count += count
		ind[f_name] = (temp, curr_count - 1)

	sorted_filenames = sorted(ind, key=ind.get)

	#Shuffle all transactions
	x = [i for i in range(total)]
	logging.critical("Shuffling")
	shuffle(x)
	#Pick the top 50000, without replacement
	sample_size_in_members = 50000
	#obtain a sorted list of indices
	sample_indices = sorted(x[:sample_size_in_members])

	result_dict = produce_sample(sample_indices, ind, sorted_filenames)

	sorted_keys = sorted(result_dict, key=result_dict.get)
	miss_count = 0
	count = 0
	with gzip.open("sample.txt.gz", "wb") as f_out:
		f_out.write(bytes(header + "\n", "UTF-8"))
		for k in sorted_keys:
			with gzip.open(k, "rb") as gzipped_input:
				logging.critical("Fetching transactions from {0}".format(k))
				is_first_line = True
				count = ind[k][0]
				result_dict[k].reverse()
				my_element = None
				for line in gzipped_input:
					if is_first_line:
						is_first_line = False
						continue
					elif my_element is None:
						if not result_dict[k]:
							break
						my_element = result_dict[k].pop()
					if count == my_element:
						line = clean_line(line)
						logging.debug("Writing sample to file.")
						f_out.write(bytes(line + "\n", "UTF-8"))
						my_element = None
					else:
						miss_count += 1
					count += 1

#MAIN PROGRAM
INPUT_PATH = sys.argv[1]
start(INPUT_PATH)

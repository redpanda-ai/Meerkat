#!/bin/python3.3
"""This module is used to create random samples from panel files."""
import glob
import gzip
import logging
import os
import sys
import random

def get_header(input_file):
	"""Gets the header from an input file."""
	logging.critical("Getting header from %s", input_file)
	with gzip.open(input_file, "rt") as gzipped_input:
		#is_first_line = True
		for line in gzipped_input:
			return line

def count_me(input_file):
	"""Counts the number of lines in a file, updates the running total"""
	logging.critical("Counting %s", input_file)
	with gzip.open(input_file, "rt") as gzipped_input:
		count = -1 #ignore the header
		for _ in gzipped_input:
			count += 1
	logging.warning("Count: {0}".format(count))
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

def start(input_path, sample_size):
	"""Runs the main program."""
	os.chdir(input_path)
	input_files = sorted(glob.glob('*.gz'))
	#ignore old sample
	input_files = [x for x in input_files if x != "rsample.txt.gz"]
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

	# get a random sample of indeces
	sample_indices = sorted(random.sample(range(total), sample_size))

	result_dict = produce_sample(sample_indices, ind, sorted_filenames)

	sorted_keys = sorted(result_dict, key=result_dict.get)
	count = 0
	lines = []

	#Build a list of lines to shuffle
	for k in sorted_keys:
		with gzip.open(k, "rt") as gzipped_input:
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
					lines.append(line)
					my_element = None
				else:
					miss_count += 1
				count += 1

	# Shuffle random sample to eliminate order bias 
	# from how we collected the list of lines
	shuffle(lines)

	#Write out the result
	with gzip.open("rsample.txt.gz", "wt") as f_out:
		f_out.write(header)
		for line in lines:
			f_out.write(line)

#MAIN PROGRAM
INPUT_PATH, SAMPLE_SIZE = sys.argv[1:3]
if len(sys.argv) != 3:
	logging.critical("Use it correctly!")
	sys.exit()
start(INPUT_PATH, SAMPLE_SIZE)

"""This module creates a random sample from a large group of gzipped files."""
#!/bin/python3.3

import glob
import gzip
import logging
import os
import sys
from random import shuffle

class Printer():
	"""Print things to stdout on one line dynamically."""
	def __init__(self, data):
		sys.stdout.write("\r\x1b[K" + data.__str__())
		sys.stdout.flush()

def clean_line(line):
	"""Trims binary junk from a gzipped line"""
	return str(line)[2:-3]

def bucket_me(input_file, shuffle_dict):
	"""Buckets an input file, placing hte result in a dictionary"""
	#print("Processing {0}".format(input_file))
	logging.critical("Bucket Processing %s", input_file)
	count = 0
	with gzip.open(input_file, "rb") as gzipped_input:
		is_first_line = True
		for line in gzipped_input:
			count += 1
			if count % CHUNK_SIZE == 0:
				output = "Bucket %d chunks completed" % (count / CHUNK_SIZE)
				Printer(output)
			line = clean_line(line)
			if is_first_line:
				header = "SHUFFLE_ID|" + line
				is_first_line = False
				continue
			shuffle_id = line.split("|")[0]
			if shuffle_id not in shuffle_dict:
				shuffle_dict[shuffle_id] = 0
			shuffle_dict[shuffle_id] += 1
	print("\n")
	make_histogram(shuffle_dict)
	return header

def get_header(input_file):
	"""Builds a header from the first line in a file."""
	logging.critical("Getting header from %s", input_file)
	with gzip.open(input_file, "rb") as gzipped_input:
		for line in gzipped_input:
			line = clean_line(line)
			header = "SHUFFLE_ID|" + line
			return header

def make_histogram(shuffle_dict):
	"""Creates a histogram of transaction buckets from the dictionary"""
	buckets = {}
	for key in shuffle_dict:
		if shuffle_dict[key] not in buckets:
			buckets[shuffle_dict[key]] = [key]
		else:
			buckets[shuffle_dict[key]].append(key)

	total = 0
	for key in buckets:
		total += len(buckets[key])

	logging.critical("\nBucket # - Members in Bucket [percent of total]")
	for sorted_key in sorted(buckets.keys()):
		logging.critical("%d - %d [%.2f]" % (sorted_key,\
			len(buckets[sorted_key]), len(buckets[sorted_key]) * 100 / total ))

def filter_me(input_file, y):
	"""Filters the a gzipped input file."""
	logging.critical("Processing %s", input_file)
	with gzip.open(input_file, "rb") as gzipped_input:
		count = 0
		for line in gzipped_input:
			count += 1
			if count % CHUNK_SIZE == 0:
				output = "Filter %d chunks completed" % (count / CHUNK_SIZE)
				Printer(output)
			line = clean_line(line)
			line_tokens = line.split("|")
			key = line_tokens[0]
			if key in y:
				y[key].append(line)
	print("\n")

def start(input_path):
	"""Runs the main program, writing results to a gzipped output
	file."""
	os.chdir(input_path)
	input_files = sorted(glob.glob('*.gz'))

	shuffle_dict, y = {}, {}
	header = get_header(input_files[0])

	[bucket_me(input_file, shuffle_dict) for input_file in input_files]

	all_members = list(shuffle_dict.keys())

	shuffle_list = [all_members[i] for i in range(len(all_members))]
	logging.critical("Shuffling")
	shuffle(shuffle_list)

	shuffle_list = shuffle_list[:SAMPLE_SIZE_IN_MEMBERS]
	for item in shuffle_list:
		y[item] = []
	[filter_me(z, y) for z in input_files]
	with gzip.open("outfile.gz", "wb") as output_file:
		output_line = bytes(header + "\n", "UTF-8")
		output_file.write(output_line)
		for c in range(len(shuffle_list)):
			count = 0
			for item in y[shuffle_list[c]]:
				count += 1
				if count % CHUNK_SIZE == 0:
					output = "Output %d chunks completed" % (count / CHUNK_SIZE)
					Printer(output)
				output_line = bytes(str(c) + "|" + str(item) + "\n", "UTF-8")
				output_file.write(output_line)
			print("\n")

#Main program
CHUNK_SIZE = 10000
SAMPLE_SIZE_IN_MEMBERS = 20000
#SAMPLE_SIZE_IN_MEMBERS = 2
start(sys.argv[1])

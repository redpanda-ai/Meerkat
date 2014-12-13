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

def bucket_me(input_file, d):
	"""Creates a buckets for the file.  This will be used for a histogram"""
	#print("Processing {0}".format(input_file))
	logging.critical("Processing %s", input_file)
	#unzipped_input = None
	count = 0
	with gzip.open(input_file, "rb") as gzipped_input:
		is_first_line = True
		for line in gzipped_input:
			count += 1
			if count % 20000 == 0:
				sys.stdout.write(".")
				sys.stdout.flush()
			line = clean_line(line)
			#print(line)
			if is_first_line:
				header = "SHUFFLE_ID|" + line
				is_first_line = False
				continue
			x = line.split("|")
			if x[0] not in d:
				d[x[0]] = 1
			else:
				d[x[0]] += 1
	make_histogram(d)
	return header

def get_header(input_file):
	"""Gets the header from an input file."""
	logging.critical("Getting header from %s", input_file)
	with gzip.open(input_file, "rb") as gzipped_input:
		#is_first_line = True
		for line in gzipped_input:
			line = clean_line(line)
			header = "SHUFFLE_ID|" + line
			return header

def make_histogram(d):
	"""Creates a histogram from the buckets."""
	buckets = {}
	for key in d:
		if d[key] not in buckets:
			buckets[d[key]] = [key]
		else:
			buckets[d[key]].append(key)

	total = 0
	for key in buckets:
		total += len(buckets[key])

	logging.critical("\nBucket # - Members in Bucket [percent of total]")
	for sorted_key in sorted(buckets.keys()):
		logging.critical("%d - %d [%.2f]" % (sorted_key,\
			len(buckets[sorted_key]), len(buckets[sorted_key]) * 100 / total ))

def filter_me(input_file, y):
	"""Filters out transactions belonging to the random sample of users."""
	logging.critical("Processing %s", input_file)
	with gzip.open(input_file, "rb") as gzipped_input:
		count = 0
		for line in gzipped_input:
			count += 1
			if count % 20000 == 0:
				sys.stdout.write(".")
				sys.stdout.flush()
			line = clean_line(line)
			x = line.split("|")
			key = x[0]
			if key in y:
				y[key].append(line)

def start(input_path):
	"""Runs the main program."""
	os.chdir(input_path)
	input_files = sorted(glob.glob('*.gz'))

	d, y = {}, {}
	header = get_header(input_files[0])
	#print(header)
	#print(input_files)
	_ = [bucket_me(x, d) for x in input_files]

	#Could add a filter here to remove where d.keys where number of records < 25
	#z = [ a for a in d.keys() if d[a] >= 25 ]
	#all_members = z

	all_members = list(d.keys())
	len_all_members = len(all_members)

	x = [all_members[i] for i in range(len_all_members)]
	logging.critical("Shuffling")
	shuffle(x)

	sample_size_in_members = 20000
	x = x[:sample_size_in_members]

	for item in x:
		y[item] = []

	_ = [filter_me(z, y) for z in input_files]

	with open("outfile", "w") as outfile:
		outfile.write(header + "\n")
		for c in range(len(x)):
			count = 0
			for item in y[x[c]]:
				count += 1
				if count % 20000 == 0:
					sys.stdout.write(".")
					sys.stdout.flush()
				line = str(c) + "|" + item + "\n"
				outfile.write(line)
				#print("%d|%s" % (c,item))

#MAIN PROGRAM
INPUT_PATH = sys.argv[1]
start(INPUT_PATH)

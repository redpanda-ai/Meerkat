import boto
import csv
import gzip
import json
import logging
import os
import re
import sys

from collections import defaultdict, OrderedDict
from boto.s3.connection import Location, Key
from meerkat.various_tools import safely_remove_file

#Usage
# python3.3 -m meerkat.tools.merge_ct_and_meerkat_panel <path_to_configuration_file>

#Example
# python3.3 -m meerkat.tools.merge_ct_and_meerkat_panel config/card_merge.json

def diff(a, b):
	b = set(b)
	return [aa for aa in a if aa not in b]

def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None

	if len(sys.argv) != 2:
		usage()
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)

	try:
		with open(sys.argv[1], encoding='utf-8') as input_file:
			params = json.loads(input_file.read())
	except IOError:
		logging.error("%s not found, aborting.", sys.argv[1])
		sys.exit()
	return params

def inter(a, b):
	b = set(b)
	return [aa for aa in a if aa in b]

def get_pending_list():
	lists = []
	my_key = re.compile(".*/([^/]+)")
	#logging.critical("-> {0}".format((PARAMS["S3"])))
	for s3_dir in PARAMS["S3"]:
		list = [j.key for j in s3_dir["s3_objects"] ]
		keys = [my_key.search(k).group(1) for k in list if my_key.search(k)]
		lists.append(keys)
	my_inter = inter(lists[0], lists[1])
	return diff(my_inter, lists[2])

def get_s3_contents(bucket_name, sub_dir):
	bucket = CONN.get_bucket(bucket_name, Location.USWest2)
	my_filter = re.compile(PARAMS["filter"])
	return [j for j in bucket.list(prefix=sub_dir) if my_filter.search(j.key)]

def set_s3():
	my_re = re.compile("S3://([^/]+)/(.*/)")
	for s3_dir in PARAMS["S3"]:
		path = s3_dir["path"]
		if my_re.match(path):
			matches = my_re.search(path)
			bucket, sub_dir = matches.group(1), matches.group(2)
			s3_dir["bucket"] = bucket
			s3_dir["sub_dir"] = sub_dir
			s3_dir["s3_objects"] = get_s3_contents(bucket, sub_dir)
			logging.warning("{0} contains {1} items.".format(sub_dir, len(s3_dir["s3_objects"])))
		else:
			logging.warning("Path is invalid, double-check your configuration file.")

def set_directories():
	for dir in PARAMS["local"]:
		path = dir["path"]
		if not os.path.exists(path):
			logging.warning("{0} does not exist.".format(path))
			os.makedirs(path)
			logging.warning("{0} created.".format(path))
		else:
			logging.debug("{0} found, continuing.".format(path))

def sort_the_file(my_file):
	logging.warning("Reading {0}".format(my_file))
	my_header = defaultdict(list)
	my_map = defaultdict(list)
	count = 0
	tock = 20000
	sort_keys = PARAMS["sort_keys"]
	with gzip.open(my_file, 'rt') as file_one:
		csv_reader = csv.reader(file_one, delimiter='|')
		first_line = True
		for row in csv_reader:
			if first_line:
				my_filter = [ idx for idx, x in enumerate(row) if x in sort_keys ]
			bar = []
			for x in my_filter:
				bar.append(row[x])
			line = ".".join(bar)
			if first_line:
				my_header[line].extend(row[0:len(row)])
				first_line = False
				continue
			if count % tock == 0:
				sys.stdout.write('.')
				sys.stdout.flush()
			if count < MAX_LINES:
				my_map[line].extend(row[0:len(row)])
				count += 1
			else:
				break
	sys.stdout.write('\n')
	sys.stdout.flush()
	logging.warning("Sorting")
	return list(my_header.values())[0], OrderedDict(sorted(my_map.items(), key=lambda t: t[0])), count

def get_columns(my_data, filter_name):
	my_filter = PARAMS[filter_name]
	#logging.warning("New filter {0}".format(new_filter))
	result = []
	try:
		result = [ my_data[1][y] for y in my_filter ]
	except:
		logging.critical("Detencted an error, skipping")
	return result

def get_new_filter(header_a, header_b):
	return [ idx for idx, x in enumerate(header_a) if x in header_b ]

def merge_the_files(args, expected_lines, remainder):
	logging.warning("Merging")

	file_name, merged_file, header_1, header_2, map_1, map_2 = args

	match_count, all_count, tick = 0, 0, 10000
	tick = int(expected_lines / 20)
	entry_a, entry_b = None, None
	PARAMS["filter_a"] = get_new_filter(header_1, header_1)
	PARAMS["filter_b"] = get_new_filter(header_2, remainder)
	with gzip.open(merged_file, 'wt') as f_out:
		header_line = "|".join(header_1) + "|".join(remainder) + "\n"
		f_out.write(header_line)
		while map_1 and map_2 and all_count < expected_lines:
			if entry_a is None:
				entry_a = map_1.popitem(last=False)
				a = entry_a[0]
			if entry_b is None:
				entry_b = map_2.popitem(last=False)
				b = entry_b[0]
			if a == b:
				match_count += 1
				part_a = get_columns(entry_a, "filter_a")
				#TODO: Apply this filter before you store it in the sorted dictionary
				part_b = get_columns(entry_b, "filter_b")
				line = "|".join(part_a) + "|".join(part_b) + "\n"
				f_out.write(line)
				entry_a = None
				entry_b = None
			all_count += 1
			if all_count % tick == 0:
				sys.stdout.write('.')
				sys.stdout.flush()

	sys.stdout.write('\n')
	sys.stdout.flush()
	logging.warning("Moving to S3")
	bucket_name = PARAMS["S3"][2]["bucket"]
	sub_dir = PARAMS["S3"][2]["sub_dir"]
	bucket = CONN.get_bucket(bucket_name, Location.USWest2)
	key = Key(bucket)
	key.key = sub_dir + file_name
	bytes_written = key.set_contents_from_filename(merged_file, encrypt_key=True, replace=True)
	logging.warning("{0} bytes written.".format(bytes_written))

def merge(file_name):
	#Abort early if the file was already completed.
	my_key = re.compile(".*/([^/]+)")
	finished_objects = PARAMS["S3"][2]["s3_objects"]
	finished_files = [my_key.search(x.key).group(1) for x in finished_objects]
	if file_name in finished_files:
		logging.critical("Skipping {0}".format(file_name))
		return

	logging.warning("Merging {0}".format(file_name))
	MAX_LINES = sys.maxsize

	#Make a sorted dictionary of the first file
	file_1 = PARAMS["local"][0]["path"] + "/" + file_name
	header_1, map_1, count_1 = sort_the_file(file_1)
	safely_remove_file(file_1)
	logging.warning("There were {0} records in the file.".format(count_1))

	#Make a sorted dictionary of the second file
	file_2 = PARAMS["local"][1]["path"] + "/" + file_name
	header_2, map_2, count_2 = sort_the_file(file_2)
	safely_remove_file(file_2)
	logging.warning("There were {0} records in the file.".format(count_2))

	#Abort if files have a different number of records
	if count_1 != count_2:
		logging.critical("ERROR! Mismatched number of lines, aborting.")
		sys.exit()

	#Merge the two files
	merged_file = PARAMS["local"][2]["path"] + "/" + file_name
	logging.warning("Files have the same number of records, proceeding")
	remainder = diff(header_2, header_1)
	args = [ file_name, merged_file, header_1, header_2, map_1, map_2 ]
	merge_the_files(args, count_1, remainder)

def process_pending_list():
	logging.warning("Processing pending list")
	my_key = re.compile(".*/([^/]+)")
	for s3_dir in PARAMS["S3"]:
		list = [j.key for j in s3_dir["s3_objects"] ]
		keys = [my_key.search(k).group(1) for k in list if my_key.search(k)]

	for x in range(0, len(PENDING)):
		_ = pull_file_from_s3(0, x)
		filename = pull_file_from_s3(1, x)
		merge(filename)

def pull_file_from_s3(i, x):
	my_key = re.compile(".*/([^/]+)")
	local = PARAMS["local"][i]["path"]
	bucket = PARAMS["S3"][i]
	s3_object = bucket["s3_objects"][x]
	filename = my_key.search(s3_object.key).group(1)
	s3_object.get_contents_to_filename(local + "/" + filename)
	logging.warning("{0} pulled from {1}".format(filename, PARAMS["S3"][i]["path"]))
	return filename


#Main program
MAX_LINES = sys.maxsize
PARAMS = initialize()
#print(PARAMS)
set_directories()
CONN = boto.connect_s3()
set_s3()
PENDING = get_pending_list()
#print(len(PENDING))
process_pending_list()



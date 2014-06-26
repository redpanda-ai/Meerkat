"""This module transforms panels built by the Clustering Tool into a format
also used by Meerkat.  It does not use Meerkat."""
import boto
import gzip
import re
import sys

from boto.s3.connection import Key, Location
from copy import deepcopy
from .various_tools import safely_remove_file

OUTPUT_FORMAT = [\
"UNIQUE_MEM_ID", "UNIQUE___BLANK_ACCOUNT_ID", "UNIQUE___BLANK_TRANSACTION_ID",\
"MEM_ID", "__BLANK_ACCOUNT_ID", "__BLANK_TRANSACTION_ID", "COBRAND_ID",\
"SUM_INFO_ID", "AMOUNT", "CURRENCY_ID", "DESCRIPTION", "TRANSACTION_DATE",\
"POST_DATE", "TRANSACITON_BASE_TYPE", "TRANSACTION_CATEGORY_ID",\
"TRANSACTION_CATEGORY_NAME", "MERCHANT_NAME", "STORE_ID", "FACTUAL_CATEGORY",\
"STREET", "CITY", "STATE", "ZIP_CODE", "WEBSITE", "PHONE_NUMBER", "FAX_NUMBER",\
"LATITUDE", "LONGITUDE", "NEIGHBOURHOOD", "TRANSACTION_ORIGIN",\
"CONFIDENCE_SCORE", "FACTUAL_ID", "FILE_CREATED_DATE"]

def begin_processing_loop(some_container):
	"""Fetches a list of input files to process from S3 and loops over them."""
	conn = boto.connect_s3()

	#Set destination details
	dst_bucket_name = "s3yodlee"
	dst_s3_path_regex = re.compile("meerkat/nullcat/" + some_container +\
	"/([^/]+)")
	#dst_local_path = "data/input/dst/"
	dst_local_path = "/mnt/ephemeral/output/"
	dst_bucket = conn.get_bucket(dst_bucket_name, Location.USWest2)

	#Get the list of completed files (already proceseed)
	#completed_list = []
	completed = {}
	for j in dst_bucket.list():
		if dst_s3_path_regex.search(j.key):
			completed[dst_s3_path_regex.search(j.key).group(1)] = j.size

	#print(completed_names)
	#print(completed_sizes)
	#sys.exit()
	#Set source details
	src_bucket_name = "s3yodlee"
	src_s3_path_regex = re.compile("ctprocessed/gpanel/" + some_container +\
	"/([^/]+)")
	#src_local_path = "data/input/src/"
	src_local_path = "/mnt/ephemeral/input/"
	src_bucket = conn.get_bucket(src_bucket_name, Location.USWest2)

	#Get list of pending files (yet to be processed)
	pending_list = []
	for k in src_bucket.list():
		if src_s3_path_regex.search(k.key):
			file_name = src_s3_path_regex.search(k.key).group(1)
			if file_name in completed:
				#Exclude files that have already been completed
				ratio = float(k.size) / completed[file_name]
				#Completed incorrectly
				if ratio >= 1.8:
					print("Completed Size, Source Size, Ratio: {0}, {1}, {2:.2f}".format(completed[file_name], k.size, ratio))
					print("Re-running {0}".format(file_name))
					pending_list.append(k)
			else:
				pending_list.append(k)
	#Reverse the pending list so that they are processed in reverse
	#chronological order
	pending_list.reverse()
	#sys.exit()
	#Loop through each file in the list of files to process
	dst_s3_path = "meerkat/nullcat/" + some_container + "/"
	for item in pending_list:
		src_file_name = src_s3_path_regex.search(item.key).group(1)
		dst_file_name = src_file_name
		print(src_file_name)
		#Copy the input file from S3 to the local file system
		item.get_contents_to_filename(src_local_path + src_file_name)
		header_name_pos, header_pos_name = get_header_dictionaries(src_file_name,\
		src_local_path)
		map_of_column_positions = get_map_of_column_positions(header_name_pos,\
		some_container)
		#Process the individual file
		process_file(src_file_name, src_local_path, dst_file_name, dst_local_path,\
		header_pos_name, map_of_column_positions, some_container)
		safely_remove_file(src_local_path + src_file_name)
		#Push the results from the local file system to S3
		dst_key = Key(dst_bucket)
		dst_key.key = dst_s3_path + src_file_name
		bytes_written = dst_key.set_contents_from_filename(dst_local_path + dst_file_name,\
		encrypt_key=True, replace=True)
		print("{0} bytes written".format(bytes_written))
		safely_remove_file(dst_local_path + dst_file_name)

def clean_line(line):
	"""Strips out the part of a binary line that is not usable"""
	return str(line)[2:-3]

def get_header_dictionaries(src_file_name, src_local_path):
	"""Pulls the header from an input file and creates the following:
		1.  A dictionary of header names and their positions
		2.  A dictionary of header positions and their names"""
	with gzip.open(src_local_path + src_file_name, "rb") as gzipped_input:
		for line in gzipped_input:
			header = clean_line(line)
			header_list = header.split("|")
			header_name_pos, header_pos_name = {}, {}
			counter = 0
			for column in header_list:
				header_name_pos[column] = counter
				header_pos_name[counter] = column
				counter += 1
			return header_name_pos, header_pos_name

def get_map_of_column_positions(header_name_pos, some_container):
	"""Builds a dictionary that maps the input header positions to the output
	header positions.  This dictionary is keyed by the column name.
	Example Key/Value: 'MEM_ID', (0,3)
	A.  The key is the name of the data column
	B.  The value is a tuple of:
		1.  The ordinal of the data column for the OUTPUT
		2.  The ordinal of the data column for the INPUT """
	output_format = get_output_format(some_container)
	map_of_column_positions = {}
	count = 0
	for column in output_format:
		if column in header_name_pos:
			map_of_column_positions[column] = (count, header_name_pos[column])
		else:
			map_of_column_positions[column] = (count, None)
		count += 1
	return map_of_column_positions

def get_output_format(some_container):
	"""Creates an output format based upon a template blended with a container
	name."""
	my_container = some_container.upper()
	output_format = [x.replace("__BLANK", my_container) for x in OUTPUT_FORMAT]
	return output_format

def process_file(src_file_name, src_local_path, dst_file_name, dst_local_path,\
header_pos_name, map_of_column_positions, container):
	""" Does the following:
		1. Takes a gzipped input file from the local file system
		2. Re-arranges the contents to meet our Meerkat output specification
		3. Stores the result in a gzipped output file which is written to the
		   local file system"""
	output_format = get_output_format(container)
	blank_result = [""] * len(output_format)
	my_container = container.upper()
	with gzip.open(src_local_path + src_file_name, "rb") as gzipped_input:
		with gzip.open(dst_local_path + dst_file_name, "wb") as gzipped_output:
			first_line = True
			for line in gzipped_input:
				#Treat the first line differently, since it is a header
				if first_line:
					first_line = False
					line = clean_line(line)
					output_line = "|".join(output_format)
					if "|GOOD_DESCRIPTION|" not in line:
						error_msg = "Error, no header found in source file."
						print(error_msg)
						line = clean_line(line)
						print("FIRST LINE:\n{0}".format(line))
						gzipped_output.write(bytes(error_msg + "\n", 'UTF-8'))
						return
				else:
					line = clean_line(line)
					split_list = line.split("|")
					result = deepcopy(blank_result)
					count = 0
					for item in split_list:
						try:
							name = header_pos_name[count]
						except:
							gzipped_output.close()
							with gzip.open(dst_local_path + dst_file_name, "wb") as gzipped_output_error:
								error_msg = "Source data is corrupt; found improperly structured record on line " + str(count)
								gzipped_output_error.write(bytes(error_msg + "\n", 'UTF-8'))
								return
						if name in map_of_column_positions:
							position = map_of_column_positions[name][0]
							result[position] = item
						count += 1
					#Add composite columns, not explicitly provided from the input
					result[1] = str(result[map_of_column_positions["COBRAND_ID"][0]])\
					+ "." + str(result[map_of_column_positions[my_container\
					+ "_ACCOUNT_ID"][0]])
					result[2] = str(result[map_of_column_positions["COBRAND_ID"][0]])\
					+ "." + str(result[map_of_column_positions[my_container\
					+ "_TRANSACTION_ID"][0]])
					#Turn the output list into a pipe-delimited string
					output_line = "|".join(result)
				#Encode the line as bytes in UTF-8 and write them to a gzipped file
				output_line = bytes(output_line + "\n", 'UTF-8')
				gzipped_output.write(output_line)

#Main program
begin_processing_loop(sys.argv[1])

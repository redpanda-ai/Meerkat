import boto
import gzip
import re
import sys

from boto.s3.connection import Key, Location, S3Connection
from copy import deepcopy
from .various_tools import safely_remove_file

OUTPUT_FORMAT = [
	"UNIQUE_MEM_ID",
	"UNIQUE___BLANK_ACCOUNT_ID",
	"UNIQUE___BLANK_TRANSACTION_ID",
	"MEM_ID",
	"__BLANK_ACCOUNT_ID",
	"__BLANK_TRANSACTION_ID",
	"COBRAND_ID",
	"SUM_INFO_ID",
	"AMOUNT",
	"CURRENCY_ID",
	"DESCRIPTION",
	"TRANSACTION_DATE",
	"POST_DATE",
	"TRANSACITON_BASE_TYPE",
	"TRANSACTION_CATEGORY_ID",
	"TRANSACTION_CATEGORY_NAME",
	"MERCHANT_NAME",
	"STORE_ID",
	"FACTUAL_CATEGORY",
	"STREET",
	"CITY",
	"STATE",
	"ZIP_CODE",
	"WEBSITE",
	"PHONE_NUMBER",
	"FAX_NUMBER",
	"LATITUDE",
	"LONGITUDE",
	"NEIGHBOURHOOD",
	"TRANSACTION_ORIGIN",
	"CONFIDENCE_SCORE",
	"FACTUAL_ID",
	"FILE_CREATED_DATE"
]

def clean_line(line):
	"""Strips out the part of a binary line that is not usable"""
	return str(line)[2:-3]

def get_output_format(container):
	"""Creates an output format based upon a template blended with a container name."""
	my_container = container.upper()
	output_format = [x.replace("__BLANK", my_container) for x in OUTPUT_FORMAT]
	return output_format

def get_output_structure(header_name_pos, container):
	"""Builds a couple of dictionaries based upon the position of columns within our header."""
	output_format = get_output_format(container)
	result_name_pos = {}
	result_pos_name = {}
	count = 0
	for column in output_format:
		if column in header_name_pos:
			result_name_pos[column] = (count, header_name_pos[column])
			result_pos_name[count] = (column, header_name_pos[column])
		else:
			result_name_pos[column] = (count, None)
			result_pos_name[count] = (column, None)
		count += 1
	return result_name_pos, result_pos_name

def connect(container):
	"""Connects to S3 to fetch a list of files to transform."""
	conn = boto.connect_s3()

	#Source details
	src_bucket_name = "s3yodlee"
	src_s3_path_regex = re.compile("ctprocessed/gpanel/" + container + "/([^/]+)")
	src_local_path = "data/input/src/"

	#Destination details
	dst_bucket_name = "s3yodlee"
	dst_s3_path_regex = re.compile("meerkat/nullcat/" + container + "/([^/]+)")
	dst_local_path = "data/input/dst/"

	#Get completed list
	dst_bucket = conn.get_bucket(dst_bucket_name, Location.USWest2)
	completed_list = []
	for j in dst_bucket.list():
		if dst_s3_path_regex.search(j.key):
			completed_list.append(dst_s3_path_regex.search(j.key).group(1))

	#Get list of files to process
	src_bucket = conn.get_bucket(src_bucket_name, Location.USWest2)
	s3_file_list = []
	for k in src_bucket.list():
		if src_s3_path_regex.search(k.key):
			file_name = src_s3_path_regex.search(k.key).group(1)
			if file_name in completed_list:
				#Remove files that have already been processed
				print("Ignoring {0}".format(file_name))
			else:
				s3_file_list.append(k)
	#Reverse the list so that they are processed in reverse chronological order
	s3_file_list.reverse()

	dst_bucket = conn.get_bucket(dst_bucket_name, Location.USWest2)
	dst_s3_path = "meerkat/nullcat/card/"
	#Loop through each file in the list of files to process
	for item in s3_file_list:
		src_file_name = src_s3_path_regex.search(item.key).group(1)
		dst_file_name = src_file_name
		print(src_file_name)
		#Copy the input file from S3 to the local file system
		item.get_contents_to_filename(src_local_path + src_file_name)
		header, header_name_pos, header_pos_name = get_header(src_file_name, src_local_path, dst_local_path)
		result_name_pos, result_pos_name = get_output_structure(header_name_pos, container)
		#Process the individual file
		process_file(src_file_name, src_local_path, dst_file_name, dst_local_path, header_pos_name, result_name_pos, container)
		safely_remove_file(src_local_path + src_file_name)
		#Push the results from the local file system to S3
		dst_key = Key(dst_bucket)
		dst_key.key = dst_s3_path + src_file_name
		dst_key.set_contents_from_filename(dst_local_path + dst_file_name, encrypt_key=True)
		safely_remove_file(dst_local_path + dst_file_name)

def process_file(src_file_name, src_local_path, dst_file_name, dst_local_path, header_pos_name, result_name_pos, container):
	""" Does the following:
		1. Takes a gzipped input file from the local file system
		2. Re-arranges the contents to meet our Meerkat output specification
		3. Stores the result in a gzipped output file which is written to the local file system"""
	output_format = get_output_format(container)
	blank_result = [""] * len(output_format)
	my_container = container.upper()
	with gzip.open(src_local_path + src_file_name, "rb") as gzipped_input:
		with gzip.open(dst_local_path + dst_file_name, "wb") as gzipped_output:
			first_line = True
			for line in gzipped_input:
				if first_line:
					first_line = False
					output_line = "|".join(output_format)
				else:
					line = clean_line(line)
					split_list = line.split("|")
					result = deepcopy(blank_result)
					count = 0
					for item in split_list:
						name = header_pos_name[count]
						if name in result_name_pos:
							position = result_name_pos[name][0]
							result[position] = item
						count += 1
					#This is not ideal, the CT tool should provide this functionality
					result[1] = str(result[result_name_pos["COBRAND_ID"][0]]) + "." + str(result[result_name_pos[my_container + "_ACCOUNT_ID"][0]])
					result[2] = str(result[result_name_pos["COBRAND_ID"][0]]) + "." + str(result[result_name_pos[my_container + "_TRANSACTION_ID"][0]])
					output_line = "|".join(result)
				output_line = bytes(output_line + "\n", 'UTF-8')
				gzipped_output.write(output_line)

def get_header(src_file_name, src_local_path, dst_local_path):
	"""Pulls the header from an input file and creates the following:
		1.  The header
		2.  A dictionary of header names and their positions
		3.  A dictionary of header positions and their names"""
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
			return line, header_name_pos, header_pos_name

#Main program
container = sys.argv[1]
connect(container)

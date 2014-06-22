import boto
import gzip
import re

from boto.s3.connection import Key, Location, S3Connection
from copy import deepcopy
from .various_tools import safely_remove_file

OUTPUT_FORMAT = [
	"UNIQUE_MEM_ID",
	"UNIQUE_CARD_ACCOUNT_ID",
	"UNIQUE_CARD_TRANSACTION_ID",
	"MEM_ID",
	"CARD_ACCOUNT_ID",
	"CARD_TRANSACTION_ID",
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
	return str(line)[2:-3]

def get_output_structure(header_name_pos):
	result_name_pos = {}
	result_pos_name = {}
	count = 0
	for column in OUTPUT_FORMAT:
		if column in header_name_pos:
			result_name_pos[column] = (count, header_name_pos[column])
			result_pos_name[count] = (column, header_name_pos[column])
		else:
			result_name_pos[column] = (count, None)
			result_pos_name[count] = (column, None)
		count += 1
	return result_name_pos, result_pos_name

def connect():
	conn = boto.connect_s3()
	bucket_src_name = "s3yodlee"
	bucket_dst_name = "s3yodlee"
#	s3_path_regex = re.compile("meerkat/test_a/([^/]+)")
	s3_path_regex = re.compile("ctprocessed/gpanel/card/([^/]+)")
	local_src_path = "data/input/src/"
	local_dst_path = "data/input/dst/"
	bucket_src = conn.get_bucket(bucket_src_name, Location.USWest2)
	s3_file_list = []
	for k in bucket_src.list():
		if s3_path_regex.search(k.key):
			s3_file_list.append(k)
	s3_file_list.reverse()
	bucket_dst = conn.get_bucket(bucket_dst_name, Location.USWest2)
	dst_s3_path = "meerkat/nullcat/card"
	for item in s3_file_list:
		src_file_name = s3_path_regex.search(item.key).group(1)
		dst_file_name = src_file_name
		print(src_file_name)
		item.get_contents_to_filename(local_src_path + src_file_name)
		header, header_name_pos, header_pos_name = get_header(src_file_name, local_src_path, local_dst_path)
		result_name_pos, result_pos_name = get_output_structure(header_name_pos)
		process_file(src_file_name, local_src_path, dst_file_name, local_dst_path, result_pos_name, header_pos_name, result_name_pos)
		safely_remove_file(local_src_path + src_file_name)
		dst_key = Key(bucket_dst)
		dst_key.key = dst_s3_path + src_file_name
		dst_key.set_contents_from_filename(local_dst_path + dst_file_name, encrypt_key=True)
		safely_remove_file(local_dst_path + dst_file_name)

def process_file(src_file_name, local_src_path, dst_file_name, local_dst_path, result_pos_name, header_pos_name, result_name_pos):
	blank_result = [""] * len(OUTPUT_FORMAT)
	with gzip.open(local_src_path + src_file_name, "rb") as gzipped_input:
		with gzip.open(local_dst_path + dst_file_name, "wb") as gzipped_output:
			first_line = True
			for line in gzipped_input:
				if first_line:
					first_line = False
					output_line = "|".join(OUTPUT_FORMAT)
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
					#This is not good, the CT tool should provide this functionality
					result[1] = str(result[result_name_pos["COBRAND_ID"][0]]) + "." + str(result[result_name_pos["CARD_ACCOUNT_ID"][0]])
					result[2] = str(result[result_name_pos["COBRAND_ID"][0]]) + "." + str(result[result_name_pos["CARD_TRANSACTION_ID"][0]])
					output_line = "|".join(result)
				output_line = bytes(output_line + "\n", 'UTF-8')
				gzipped_output.write(output_line)

def get_header(src_file_name, local_src_path, local_dst_path):
	with gzip.open(local_src_path + src_file_name, "rb") as gzipped_input:
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

connect()

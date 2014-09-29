import boto
import gzip
import re
import sys

from boto.s3.connection import Key, Location

def begin_processing_loop(some_container, day_of_month):
	"""Fetches a list of input files to process from S3 and loops over them."""
	conn = boto.connect_s3()

	#Set destination details
	dst_bucket_name = "yodleeprivate"
	#dst_s3_path_regex = re.compile("panels/meerkat/" + some_container +\
	#"/([^/]+)")
	dst_s3_path_regex = re.compile("panels/meerkat/" + some_container +\
	"/(.*" + day_of_month + "_[^/]+)")
	#dst_local_path = "data/input/dst/"
	dst_local_path = "/mnt/ephemeral/output/"
	dst_bucket = conn.get_bucket(dst_bucket_name, Location.USWest2)

	#Get the list of completed files (already proceseed)
	#completed_list = []
	completed = {}
	for j in dst_bucket.list():
		if dst_s3_path_regex.search(j.key):
			completed[dst_s3_path_regex.search(j.key).group(1)] = j.size

	#print(completed)
	#print(completed_sizes)
	#sys.exit()
	#Set source details
	src_bucket_name = "yodleeprivate"
	src_s3_path_regex = re.compile("ctprocessed/gpanel/" + some_container +\
	"/(.*" + day_of_month + "_[^/]+)")
	#src_local_path = "data/input/src/"
	src_local_path = "/mnt/ephemeral/input/"
	src_bucket = conn.get_bucket(src_bucket_name, Location.USWest2)

	#Get list of pending files (yet to be processed)
	pending_list = []
	for k in src_bucket.list():
		if src_s3_path_regex.search(k.key):
			#DEBUG
			print(k.key)
			file_name = src_s3_path_regex.search(k.key).group(1)
			if file_name in completed:
				#Exclude files that have already been completed
				ratio = float(k.size) / completed[file_name]
				#Completed incorrectly
				if ratio >= 1.8:
					print("Completed Size, Source Size, Ratio: {0}, {1}, {2:.2f}".format(completed[file_name], k.size, ratio))
					print("Re-running {0}".format(file_name))
					pending_list.append((k, k.size))
			else:
				pending_list.append((k, k.size))
	#Reverse the pending list so that they are processed in reverse
	#chronological order
	pending_list.reverse()
	dst_s3_path = "panels/meerkat/" + some_container + "/"
	#pattern = re.compile("^201[34].*$")
	pattern1 = re.compile("^201308.*$")
	pattern2 = re.compile("^201309.*$")
	pattern3 = re.compile("^201310.*$")
	pattern4 = re.compile("^201311.*$")
	pattern5 = re.compile("^201312.*$")
	pattern6 = re.compile("^2014.*$")

	large_files, large_sizes = {}, {}
	for item, item_size in pending_list:
		src_file_name = src_s3_path_regex.search(item.key).group(1)
		dst_file_name = src_file_name
		if pattern1.search(src_file_name) or pattern2.search(src_file_name) or pattern3.search(src_file_name) or pattern4.search(src_file_name) or pattern5.search(src_file_name) or pattern6.search(src_file_name):
			#print("{0} {1} {2}".format(some_container, src_file_name[0:8], int(item_size / 100000000) ))
			large_files[item.key] = int(item_size / 100000000)
			if item_size not in large_sizes:
				large_sizes[item_size] = []
			large_sizes[item_size].append((item.key))
	return large_files, large_sizes

def split_files(sizes):
	for key in sorted(sizes.keys(), reverse=True):
		print(sizes[key])

my_files, my_sizes = begin_processing_loop(sys.argv[1], sys.argv[2])
split_files(my_sizes)


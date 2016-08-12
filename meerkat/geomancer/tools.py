import re
import queue
import datetime
import argparse
import csv
import json
import logging
import sys
import yaml
import shutil
import pandas as pd
import threading
import time

from timeit import default_timer as timer

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('tools')

class ThreadProducer(threading.Thread):
	def __init__(self, param):
		threading.Thread.__init__(self)
		self.param = param
		self.geo_df = None

	def run(self):
		input_file = self.param["input_file"]
		csv_kwargs = self.param["csv_kwargs"]
		count = 0
		for chunk in pd.read_csv(input_file, **csv_kwargs):
			self.param["data_queue"].put(chunk)
			count += 1
			logger.info("Populating data queue {0}".format(str(count)))
		self.param["data_queue_populated"] = True
		logger.info("data queue is populated, data queue size: {0}".format(self.param["data_queue"].qsize()))


class ThreadConsumer(threading.Thread):
	def __init__(self, thread_id, param):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.param = param
		self.param["consumer_queue"].put(self.thread_id)

	def run(self):
		param = self.param
		consumer_queue = param["consumer_queue"]
		while True:
			logger.info("data_queue_populated: {0}, data_queue empty {1}".format(param["data_queue_populated"],
				param["data_queue"].empty()))
			if param["data_queue_populated"] and param["data_queue"].empty():
				#Remove yourself from the consumer queue
				logger.info("Removing consumer thread.")
				param["consumer_queue"].get()
				logger.info("Notifying task done.")
				param["consumer_queue"].task_done()
				logger.info("Consumer thread {0} finished".format(str(self.thread_id)))
				break
			chunk = param["data_queue"].get()
			logger.info("consumer thread: {0}; data queue size: {1}, consumer queue size {2}".format(str(self.thread_id),
				 param["data_queue"].qsize(), param["consumer_queue"].qsize()))

			param["chunk_num"] += 1
			if param["chunk_num"] % 10 == 0:
				logger.info("Processing chunk {0:07d} of {1:07d}, {2:>6} target merchants found.".format(param["chunk_num"],
					param["num_chunks"], len(param["dict_of_df_lists"].keys())))
				elapsed = timer() - param["start"]
				remaining = param["num_chunks"] - param["chunk_num"]
				completion = float(param["chunk_num"]) / param["num_chunks"] * 100
				chunk_rate = float(param["chunk_num"]) / elapsed
				remaining_time = float(remaining) / chunk_rate
				#Log our progress
				logger.info(param["log_string"].format(
					str(datetime.timedelta(seconds=elapsed))[:-7],
					str(datetime.timedelta(seconds=remaining_time))[:-7],
					completion, chunk_rate))

			if param["activate_cnn"]:
				transactions = chunk.to_dict('records')
				enriched = param["classifier"](transactions, doc_key='DESCRIPTION_UNMASKED',
					label_key=param["groupby_name"])
				chunk = pd.DataFrame(enriched)

			grouped = chunk.groupby(param["groupby_name"], as_index=False)
			groups = dict(list(grouped))
			my_keys = groups.keys()
			for key in my_keys:
				if key in self.param["target_merchant_list"]:
					if key not in param["dict_of_df_lists"]:
						logger.info("***** Discovered {0:>30} ********".format(key))
						param["dict_of_df_lists"][key] = []
					param["dict_of_df_lists"][key].append(groups[key])
			time.sleep(0.1)
			param["data_queue"].task_done()

def start_producers(param):
	"""Fill me in later"""
	logger.info("start producer")
	producer = ThreadProducer(param)
	producer.start()

def start_consumers(param):
	"""Fill me in later"""
	for i in range(8):
		logger.info("start consumer: {0}".format(str(i)))
		consumer = ThreadConsumer(i, param)
		consumer.setDaemon(True)
		consumer.start()

def get_grouped_dataframes(input_file, groupby_name, target_merchant_list, **csv_kwargs):
	"""Generate a dataframe which is a subset of the input_file grouped by merchant."""
	logger.info("Constructing dataframe from file.")
	activate_cnn = csv_kwargs.get("activate_cnn", False)

	if "activate_cnn" in csv_kwargs:
		del csv_kwargs["activate_cnn"]

	#Here are the target merchants
	#create a list of dataframe groups, filtered by merchant name
	dict_of_df_lists = {}
	chunk_num = 0
	if activate_cnn:
		from ..classification.load_model import get_tf_cnn_by_name as get_classifier
		classifier = get_classifier(csv_kwargs.get("cnn", "bank_merchant"))
		del csv_kwargs["cnn"]
	num_chunks = int(sum(1 for line in open(input_file)) / csv_kwargs["chunksize"])
	start = timer()
	log_string = "Taken: {0} , ETA: {1}, Competion: {2:.2f}%, Chunks/sec: {3:.2f}"
	param = {
		"activate_cnn": activate_cnn,
		"chunk_num": 0,
		"consumer_queue": queue.Queue(),
		"csv_kwargs": csv_kwargs,
		"data_queue": queue.Queue(),
		"data_queue_populated": False,
		"dict_of_df_lists": {},
		"groupby_name": groupby_name,
		"input_file": input_file,
		"num_chunks": num_chunks,
		"target_merchant_list": target_merchant_list
	}

	if activate_cnn:
		param["classifier"] = classifier
	param["start"] = start
	param["log_string"] = log_string

	start_producers(param)
	start_consumers(param)
	param["consumer_queue"].join()
	param["data_queue"].join()

	#Show what you found and did not find
	dict_of_df_lists = param["dict_of_df_lists"]
	merchants_found = dict_of_df_lists.keys()
	found_list = list(merchants_found)
	logger.info("Target Merchants: {0}".format(target_merchant_list))
	missing_list = list(set(target_merchant_list) - set(found_list))
	for item in found_list:
		logger.info("Found {0:>49}".format(item))
	logger.info("Found list {0}".format(found_list))
	logger.info("Missing list {0}".format(missing_list))
	for item in missing_list:
		logger.warning("Not Found {0:>42}".format(item))
	#Merge them together
	for key in merchants_found:
		dict_of_df_lists[key] = pd.concat(dict_of_df_lists[key], ignore_index=True)
		#Do some pre-processing
		logger.info("Preprocessing dataframe for {0:>27}".format(key))

	return dict_of_df_lists

def remove_special_chars(input_string):
	"""Remove special characters in the input strint"""
	return re.sub(r"[ |\-|'|.|&]", r'', input_string)

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file")
	parser.add_argument("--subset", default="")
	return parser.parse_args(args)

def deduplicate_csv(input_file, subset, inplace, **csv_kwargs):
	"""This function de-deduplicates transactions in a csv file"""
	read = csv_kwargs["read"]
	to = csv_kwargs["to"]
	df = pd.read_csv(input_file, **read)
	original_len = len(df)

	if subset == "":
		unique_df = df.drop_duplicates(keep="first", inplace=inplace)
	else:
		unique_df = df.drop_duplicates(subset=subset, keep="first", inplace=inplace)

	if inplace:
		logging.info("reduced {0} duplicate transactions".format(original_len - len(df)))
		df.to_csv(input_file, **to)
		logging.info("csv files with unique {0} transactions saved to: {1}".format(len(df), input_file))
	else:
		logging.info("reduced {0} duplicate transactions".format(len(df) - len(unique_df)))
		last_slosh = input_file.rfind("/")
		output_file = input_file[: last_slosh + 1] + 'deduplicated_' + input_file[last_slosh + 1 :]
		unique_df.to_csv(output_file, **to)
		logging.info("csv files with unique {0} transactions saved to: {1}".format(len(unique_df), output_file))

def get_geo_dictionary(input_file):
	"""This function takes a csv file containing city, state, and zip and creates
	a dictionary."""
	my_dict = {}
	with open(input_file) as infile:
		for line in infile:
			parts = line.split("\t")
			city = parts[2].upper()
			state = parts[4].upper()
			zipcode = parts[1]
			if state not in my_dict:
				my_dict[state] = {}
			if city not in my_dict[state]:
				my_dict[state][city] = [zipcode]
			else:
				my_dict[state][city].append(zipcode)

	my_json = json.dumps(my_dict, sort_keys=True, indent=4, separators=(',', ': '))
	return my_json

def copy_file(input_file, directory):
	"""This function moves uses Linux's 'cp' command to copy files on the local host"""
	logging.info("Copy the file {0} to directory: {1}".format(input_file, directory))
	shutil.copy(input_file, directory)

if __name__ == "__main__":
	logging.critical("Do not run this module from the command line.")

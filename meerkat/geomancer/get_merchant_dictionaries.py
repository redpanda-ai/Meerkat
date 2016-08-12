import argparse
import csv
import datetime
import inspect
import json
import os
import pandas as pd
import json
import queue
import threading
import datetime
import time
import sys
import logging
import yaml

from functools import reduce
from timeit import default_timer as timer
from .tools import remove_special_chars

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('get_merchant_dictionaries')

def preprocess_dataframe(df):
	"""Fix up some of the data in our dataframe."""
	capitalize_word = lambda x: x.upper()
	df["state"] = df["state"].apply(capitalize_word)
	df["city"] = df["city"].apply(capitalize_word)
	df["city"] = df["city"].apply(expand_abbreviations)

def get_merchant_dataframes(input_file, groupby_name, target_merchant_list, **csv_kwargs):
	"""Generate a dataframe which is a subset of the input_file grouped by merchant."""
	logger.info("Constructing dataframe from file.")
	activate_cnn = csv_kwargs.get("activate_cnn", False)
	if "activate_cnn" in csv_kwargs:
		del csv_kwargs["activate_cnn"]

	#Here are the target merchants
	#create a list of dataframe groups, filtered by merchant name
	dict_of_df_lists = {}
	chunk_num = 0
	#logger.info("Filtering by the following merchant: {0}".format(merchant))
	#for chunk in pd.read_csv(input_file, chunksize=chunksize, error_bad_lines=False,
	#	warn_bad_lines=True, encoding='utf-8', quotechar='"', na_filter=False, sep=sep):
	if activate_cnn:
		from ..classification.load_model import get_tf_cnn_by_name as get_classifier
		classifier = get_classifier("bank_merchant")
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

def start_producers(param):
	logger.info("start producer")
	producer = ThreadProducer(param)
	producer.start()

def start_consumers(param):
	for i in range(8):
		logger.info("start consumer: {0}".format(str(i)))
		consumer = ThreadConsumer(i, param)
		consumer.setDaemon(True)
		consumer.start()

"""
def parse_arguments(args):
	module_path = inspect.getmodule(inspect.stack()[1][0]).__file__
	default_path = module_path[:module_path.rfind("/") + 1] + "dictionaries"
	parser = argparse.ArgumentParser()

	parser.add_argument("dry_run", default="False", choices=["True", "False"])
	parser.add_argument("target_merchants", default=["Starbucks"])
	parser.add_argument("--target_merchant_list", type=list, default=[])
	parser.add_argument("--merchant", default="Starbucks")
	parser.add_argument("--filepath", default=default_path)

	my_args = parser.parse_args(args)
	my_args.target_merchant_list = my_args.target_merchants[2:-2].split(",")
	my_args.target_merchant_list = [ x.strip('" ') for x in my_args.target_merchant_list ]
	return my_args
"""
def merge(a, b, path=None):
	"""Useful function to merge complex dictionaries, courtesy of:
	https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge
	I added logic for handling merge conflicts so that new values overwrite old values
	but throw warnings.
	"""
	if path is None:
		path = []
	for key in b:
		if key in a:
			if isinstance(a[key], dict) and isinstance(b[key], dict):
				merge(a[key], b[key], path + [str(key)])
			elif a[key] == b[key]:
				pass
			else:
				#Merge conflict, in our case old beats new
				logger.warning("Conflict at %s" % ".".join(path + [str(key)]))
				logger.warning("A key {0}".format(a[key]))
				logger.warning("B key {0}".format(b[key]))
				logger.warning("Conflict in dicts, keeping older value.")
		else:
			a[key] = b[key]
	return a

def expand_abbreviations(city):
	"""Turns abbreviations into their expanded form."""
	maps = {
		"E. ": "EAST ", "W. ": "WEST ", "N. ": "NORTH ", "S. ": "SOUTH ",
		"ST. ": "SAINT ", "ST ": "SAINT ", "FT. ": "FORT ", "FT ": "FORT "
	}
	for abbr in maps:
		if city.startswith(abbr):
			city = city.replace(abbr, maps[abbr])
			break
	return city

class Worker:
	"""Contains methods and data pertaining to the creation and retrieval of merchant dictionaries"""
	def __init__(self, config):
		"""Constructor"""
		self.config = config
		module_path = inspect.getmodule(inspect.stack()[1][0]).__file__
		self.filepath = module_path[:module_path.rfind("/") + 1] + "merchants"

	def get_store_dictionaries(self, df):
		"""Writes out two store dictionaries"""
		logger.debug("Generating store dictionaries.")
		#Use only the "store_number", "city", and "state" columns
		slender_df = df[["store_number", "city", "state"]]
		store_dict_1, store_dict_2 = {}, {}
		my_stores = slender_df.set_index("store_number").T.to_dict('list')
		#my_stores = {str(k):str(v) for k,v in my_stores.items()}
		#Split the store_id dicts
		for key in my_stores.keys():
			key = str(key)
			#If each key cannot be split by a dash, return the full my_stores_dictionary
			if key.count("-") == 0:
				self.dump_pretty_json_to_file(my_stores, "store_id_1.json")
				self.dump_pretty_json_to_file(my_stores, "store_id_2.json")
				return my_stores, my_stores
			#Otherwise, build a split dictionary
			key_1, key_2 = key.split("-")
			store_dict_1[key_1] = my_stores[key]
			store_dict_2[key_2] = my_stores[key]
		#Dump the store_id dictionaries
		self.dump_pretty_json_to_file(store_dict_1, "store_id_1.json")
		self.dump_pretty_json_to_file(store_dict_2, "store_id_2.json")
		#Return the dictionaries
		return store_dict_1, store_dict_2

	def get_unique_city_dictionaries(self, df):
		"""Constructs a dictionary using unique city names as keys."""
		logger.debug("Generating unique_city dictionaries for {0}".format(self.config["merchant"]))
		# Create the unique_city_state dictionary
		grouped_city = self.geo_df.groupby('city', as_index=True)
		groups_city = dict(list(grouped_city))
		unique_city_state = {}
		for city, group in groups_city.items():
			states = group.state.unique()
			if len(states) == 1:
				unique_city_state[city.upper()] = states[0].upper()
		# Write the unique_city_state dictionary to json file
		merchant = self.config["merchant"]
		self.dump_pretty_json_to_file(unique_city_state, "unique_city_state.json")
		# Create the unique_city list
		unique_city = list(unique_city_state.keys())
		# Write the unique_city list to json file
		self.dump_pretty_json_to_file(unique_city, "unique_city.json")
		return unique_city_state, unique_city

	def get_geo_dictionary(self, df):
		"""Generates three merchant dictionaries and writes them as JSON files"""
		merchant = self.config["merchant"]
		logger.debug("Generating geo dictionaries for '{0}'".format(self.config["merchant"]))
		#Create a geo-dictionary, using only "state", "city", and "zip_code"
		geo_df = df[["state", "city", "zip_code"]]
		grouped = geo_df.groupby(['state', 'city'], as_index=True)
		geo_dict = {}
		for name, group in grouped:
			state, city = name
			state, city = state.upper(), city.upper()
			if state not in geo_dict:
				geo_dict[state] = {}
			if city not in geo_dict[state]:
				geo_dict[state][city] = []
			for item in group["zip_code"]:
				item = str(item)
				item = item[:5]
				if item not in geo_dict[state][city]:
					geo_dict[state][city].append(item)
		#Write the geo-dictionary
		self.dump_pretty_json_to_file(geo_dict, "geo.json")
		#Return the dataframe
		return geo_df

	def main_process(self):
		"""This is where it all happens."""
		csv_kwargs = { "chunksize": 1000, "error_bad_lines": False, "warn_bad_lines": True,
			"encoding": "utf-8", "quotechar" : '"', "na_filter" : False, "sep": "," }
		merchant_dataframes = get_merchant_dataframes("meerkat/geomancer/data/agg_data/All_Merchants.csv",
			"list_name", self.config["target_merchant_list"], **csv_kwargs)
		merchants = sorted(list(merchant_dataframes.keys()))
		for merchant in merchants:
			self.config["merchant"] = remove_special_chars(merchant)
			df = merchant_dataframes[merchant]
			preprocess_dataframe(df)
			logger.info("***** Processing {0:>29} ********".format(merchant))
			self.setup_directories()
			store_dict_1, store_dict_2 = self.get_store_dictionaries(df)
			self.geo_df = self.get_geo_dictionary(df)
			unique_city_state, unique_city = self.get_unique_city_dictionaries(df)
			# Let's also get the question bank

	def setup_directories(self):
		"""This creates the directories on the local file system, if needed."""
		if self.config["dry_run"] == "False":
			output_directory = self.filepath + "/" + self.config["merchant"]
			logger.debug("Confirming output directory at {0}".format(output_directory))
			os.makedirs(output_directory, exist_ok=True)
		else:
			logger.info("No need for output directory for {0}".format(self.config["merchant"]))

	def dump_pretty_json_to_file(self, new_object, filename):
		"""Dumps a pretty-printed JSON object to the file provided."""
		src_object = {}
		dst_object = {}
		full_path = self.filepath + "/" + self.config["merchant"] + "/" + filename
		try:
			with open(full_path, "r") as infile:
				src_object = json.load(infile)
		except IOError as e:
			logger.debug("No pre-existing object, which is fine.")
		#Merge original and new new_object
		if isinstance(new_object, dict):
			dst_object = reduce(merge, [src_object, new_object])
			dst_object = {str(k): v for k, v in dst_object.items()}
		elif isinstance(new_object, list):
			dst_object = list(set().union(new_object, src_object))
		else:
			logger.critical("It's neither a list nor a dictionary, aborting.")
			sys.exit()
		#Dump, if necessary
		log_write = "'" + self.config["merchant"] + "/" + filename + "'"
		if self.config["dry_run"] == "False":
			logger.info("Writing {0}".format(log_write))
			with open(full_path, "w") as outfile:
				json.dump(dst_object, outfile, sort_keys=True, indent=4, separators=(',', ': '))
		else:
			logger.info("Not Writing {0}".format(log_write))


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


if __name__ == "__main__":
	logger.critical("You cannot run this from the command line, aborting.")


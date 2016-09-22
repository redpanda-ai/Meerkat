import csv
import json
import logging
import numpy as np
import os
import os.path
import pandas as pd
import requests
import sys
import yaml
import boto3

from .tools import get_top_merchant_names
from .geomancer_module import GeomancerModule
from .interrogate import get_existing_projects
from meerkat.various_tools import load_params

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('taskrun_collector')

def get_taskrun_df(server, map_id_to_name, map_name_to_id):
	"""Builds a pandas dataframe containing task_funs for the indicated project."""
	offset, limit, port = 0, 100, "12000"
	remaining_data = True
	prefix = "http://" + server + ":" + port + "/api/"

	results, comments = {}, {}
	result_keys = ["locked", "city", "state", "zipcode", "streetaddress", "storenumber", "not_in_us"]
	id_keys = ["user_id", "task_id"]

	while remaining_data:
		address = prefix + "taskrun?limit=" + str(limit) + "&offset=" + str(offset)
		my_json = requests.get(address).json()
		offset = offset + limit
		logger.info("offset: {0}".format(offset))
		if not my_json:
			logger.warning("Taskrun collection complete")
			remaining_data = False
			if len(results) == 0:
				logger.info("Not found taskrun result")
			else:
				for key in results:
					logger.info("Adding {0} to results".format(key))
					results[key] = pd.DataFrame(results[key])
			if len(comments) == 0:
				logger.info("Not found comment result")
			else:
				for key in comments:
					logger.info("Adding {0} to comments".format(key))
					logger.info("{0}, {1}".format(key, comments[key]))
					comments[key] = pd.DataFrame(comments[key])
			return results, comments

		for item in my_json:
			if item["project_id"] in map_id_to_name:
				my_info = item["info"]
				project_id = item["project_id"]
				project_name = map_id_to_name[project_id]
				logger.info("Project ID: {0}, Project Name: {1}".format(project_id, project_name))
				if project_name not in results:
					logger.info("Found taskrun results for {0}".format(project_name))
					results[project_name] = {}
					comments[project_name] = {}
					for key in result_keys + id_keys:
						results[project_name][key] = []
					for key in result_keys + id_keys:
						comments[project_name][key] = []

				has_comment = False
				if "locked" in item["info"] and not item["info"]["locked"]:
					has_comment = True
				if has_comment:
					for key in result_keys:
						comments[project_name][key].append(item["info"][key])
					for key in id_keys:
						comments[project_name][key].append(item[key])
				else:
					for key in result_keys:
						results[project_name][key].append(item["info"][key])
					for key in id_keys:
						results[project_name][key].append(item[key])

def get_task_question_by_id(server, task_id):
	"""Send request to pybossa server and get task question by task id"""
	port = "12000"
	prefix = "http://" + server + ":" + port + "/api/"

	address = prefix + "task/" + str(task_id)
	my_json = requests.get(address).json()
	if "info" in my_json and "question" in my_json["info"]:
		return my_json["info"]["question"]
	else:
		return ""

def process_taskrun_dfs(dfs, server, redundancy, label_type):
	for key in dfs:
		slim_df = dfs[key]
		logger.info("Processing taskrun dataframes for Project: {0}".format(key))

		component_dataframes = []
		grouped = slim_df.groupby(["task_id"], as_index=True)
		for name, group in grouped:
			if label_type in ["unanimous", "nonunanimous"]:
				original_count = len(group)
				#dedup = group.drop_duplicates(subset=["city", "state", "zipcode", "not_in_us", "storenumber", "streetaddress"])
				dedup = group.drop_duplicates(subset=["city", "state", "not_in_us"])
				if label_type == "unanimous" and len(dedup) == 1 and original_count == redundancy:
					component_dataframes.append(dedup)
				elif label_type == "nonunanimous" and len(dedup) > 1 and original_count == redundancy:
					component_dataframes.append(group)
			else:
				component_dataframes.append(group)

		count_of_taskrun_df = len(component_dataframes)
		logger.info("Count of taskrun {0}: {1}".format(label_type, count_of_taskrun_df))

		questions = []
		for df in component_dataframes:
			for index, row in df.iterrows():
				question = get_task_question_by_id(server, int(row["task_id"]))
				questions.append(question)
		aligned_df = pd.concat(component_dataframes, axis=0)
		aligned_df["question"] = questions
		logger.info("labeled task {0}:\n{1}\n".format(label_type, aligned_df))

		bank_or_card = key.split("_")[1]
		merchant = key.split("_")[2]
		target_path = "meerkat/geomancer/merchants/" + merchant + "/"
		os.makedirs(target_path, exist_ok=True)
		aligned_df.to_csv(target_path + bank_or_card + "_taskrun_" + label_type + ".csv", sep="\t", index=False)

class Worker(GeomancerModule):
	"""Contains methods and data pertaining to the creation and retrieval of AggData files"""
	name = "taskrun_collector"
	def __init__(self, common_config, config):
		"""Constructor"""
		super(Worker, self).__init__(common_config, config)

	def main_process(self):
		"""Execute the main program"""
		bank_or_card = self.common_config["bank_or_card"]

		server, apikey = self.common_config["server"], self.common_config["apikey"]
		existing_projects = get_existing_projects(server, apikey)
		logger.info("Existing projects are: {0}".format(existing_projects))

		base_dir = "meerkat/geomancer/merchants/"
		target_merchant_list = self.common_config["target_merchant_list"]
		top_merchants = get_top_merchant_names(base_dir, target_merchant_list)

		#existing_projects = ["Geomancer_card_Target"]
		bank_or_card = "card"
		top_merchants = ["Target"]
		map_id_to_name, map_name_to_id = {}, {}
		for merchant in top_merchants:
			project_name = "Geomancer_" + bank_or_card + "_" + merchant
			if project_name in existing_projects:
				project_id = existing_projects[project_name]
				map_id_to_name[project_id] = project_name
				map_name_to_id[project_name] = project_id

		logger.info("map_id_to_name: {0}".format(map_id_to_name))
		logger.info("map_name_to_id: {0}".format(map_name_to_id))

		results, comments = get_taskrun_df(server, map_id_to_name, map_name_to_id)
		redundancy = 2
		process_taskrun_dfs(results, server, redundancy, "unanimous")
		process_taskrun_dfs(results, server, redundancy, "nonunanimous")
		process_taskrun_dfs(comments, server, redundancy, "comment")

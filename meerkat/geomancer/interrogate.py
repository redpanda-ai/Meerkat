import sys
import pandas as pd
import requests
import logging
import yaml
import time
import os

from .pybossa.build_pybossa_project import get_top_merchant_names, add_tasks

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('interrogate')

def get_task_df(server, map_id_to_name, map_name_to_id):
	"""Builds a pandas dataframe containing task_funs for the indicated project."""
	offset, limit, port = 0, 100, "12000"
	remaining_data = True
	prefix = "http://" + server + ":" + port + "/api/"

	dfs = {}
	while remaining_data:
		address = prefix + "task?limit=" + str(limit) + "&offset=" + str(offset)
		my_json = requests.get(address).json()
		offset = offset + limit
		logger.info("offset: {0}".format(offset))
		if not my_json:
			logger.warning("Task collection complete")
			remaining_data = False
			for key in dfs:
				logger.info("Adding {0} to dfs".format(key))
				dfs[key] = pd.DataFrame(dfs[key])
			return dfs
		for item in my_json:
			#logger.info("item: {0}".format(item))
			if item["project_id"] in map_id_to_name:
				my_info = item["info"]
				project_id = item["project_id"]
				project_name = map_id_to_name[project_id]
				logger.info("Project ID: {0}, Project Name: {1}".format(project_id, project_name))
				if project_name not in dfs:
					logger.info("Found tasks for {0}".format(project_name))
					dfs[project_name] = {"question": []}
				dfs[project_name]["question"].append(item["info"]["question"])

def mix_dataframes(df_1, df_2, group_size):
	"""Mix two dataframes by group size"""
	mix_df = pd.concat([df_1, df_2]).reset_index(drop=True)
	mix_df_gpby = mix_df.groupby(list(mix_df.columns))

	mixed_set = [x[0] for x in mix_df_gpby.groups.values() if len(x) == group_size]
	mixed_set = mix_df.reindex(mixed_set)
	return mixed_set

def get_new_tasks(old_df, new_df):
	"""Find new dataframes"""
	set_1 = mix_dataframes(old_df, new_df, 1)
	set_2 = mix_dataframes(old_df, set_1, 1)
	set_3 = mix_dataframes(set_1, set_2, 2)
	logger.info("new tasks df :")
	logger.info(set_3)
	return set_3

def get_existing_projects(server, apikey):
	"""Get a list of existing pybossa projects"""
	port = "12000"
	address = "http://" + server + ":" + port + "/api/project?api_key=" + apikey
	my_json = requests.get(address).json()
	short_names = {}
	for item in my_json:
		short_names[item["short_name"]] = item["id"]
	return short_names

class Worker:
	"""Contains methods and data pertaining to the creation and retrieval of AggData files"""
	def __init__(self, config):
		"""Constructor"""
		self.config = config

	def main_process(self):
		"""Execute the main program"""
		server, apikey = self.config["server"], self.config["apikey"]
		existing_projects = get_existing_projects(server, apikey)
		logger.info("Existing projects are: {0}".format(existing_projects))

		base_dir = "meerkat/geomancer/merchants/"
		top_merchants = get_top_merchant_names(base_dir)
		if len(top_merchants) == 0:
			logger.critical("There are no top merchants, aborting.")
			sys.exit()
		#Ensure that all top merchants have a pybossa directory
		for merchant in top_merchants:
			path = "meerkat/geomancer/merchants/" + merchant + "/pybossa_project/" +\
				self.config["bank_or_card"]
			if not os.path.exists(path):
				logger.critical("Cannot proceed, path not found at: {0}".format(path))
				sys.exit()
		logger.info("Top merchants are: {0}".format(top_merchants))

		bank_or_card = self.config["bank_or_card"]
		map_id_to_name, map_name_to_id = {}, {}
		for merchant in top_merchants:
			project_name = "Geomancer_" + bank_or_card + "_" + merchant
			if project_name in existing_projects:
				project_id = existing_projects[project_name]
				map_id_to_name[project_id] = project_name
				map_name_to_id[project_name] = project_id

		logger.info("map_id_to_name: {0}".format(map_id_to_name))
		logger.info("map_name_to_id: {0}".format(map_name_to_id))

		dfs = get_task_df(server, map_id_to_name, map_name_to_id)
		logger.warning("dfs: {0}".format(dfs))

		csv_kwargs = { "usecols": ["DESCRIPTION_UNMASKED"], "error_bad_lines": False, "warn_bad_lines": True,
			"encoding": "utf-8", "quotechar" : '"', "na_filter" : False, "sep": "," }

		for project_name in dfs:
			logger.info("Interrogating {0}".format(project_name))
			old_df = dfs[project_name]
			old_df = old_df.rename(columns = {'question': 'DESCRIPTION_UNMASKED'})
			logger.info("old_df: ")
			logger.info(old_df)

			merchant = project_name[len("Geomancer_") + len(bank_or_card) + 1:]
			new_tasks_file = base_dir + merchant + "/" + bank_or_card  +"_tasks.csv"
			new_df = pd.read_csv(new_tasks_file, **csv_kwargs)
			logger.info("new_df: ")
			logger.info(new_df)
			new_tasks_df = get_new_tasks(old_df, new_df)

			if new_tasks_df.empty:
				logger.info("No new tasks for {0}".format(project_name))
				continue

			tasks_file = base_dir + merchant + "/pybossa_project/" + bank_or_card + "/tasks.csv"
			new_tasks_df.to_csv(tasks_file, header=["question"], index=False)
			logger.info("Save new tasks dataframe to {0}".format(tasks_file))
			project_json_file = base_dir + merchant + "/pybossa_project/" + bank_or_card + "/project.json"
			add_tasks(server, apikey, project_json_file, tasks_file)
			logger.info("Add new tasks to {0}".format(project_name))

if __name__ == "__main__":
	logger.critical("This module cannot be run from the command line.")
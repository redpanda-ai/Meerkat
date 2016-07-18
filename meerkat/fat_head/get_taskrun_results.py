import argparse
import csv
import json
import logging
import numpy as np
import pandas as pd
import requests
import time

def parse_arguments():
	"""Parses arguments from the command-line"""
	parser = argparse.ArgumentParser()
	parser.add_argument("host", help="IP or name of the server hosting the PyBossa project")
	parser.add_argument("short_name", help="Short name for the project")
	parser.add_argument("offset", help="Task id offset", type=int, default=1)
	return parser.parse_args()

def get_project_id(args):
	"""Transalates the 'short_name' for a project into its 'project_id'"""
	port = "12000"
	prefix = "http://" + args.host + ":" + port + "/api/"
	address = prefix + "project"
	my_json = requests.get(address).json()
	for item in my_json:
		if item["short_name"] == args.short_name:
			return item["id"]
	raise Exception('Invalid short_name')

def get_task_run_df(args, project_id):
	"""Builds a pandas dataframe containing task_funs for the indicated project."""

	offset, limit, port = 0, 100, "12000"
	remaining_data = True
	prefix = "http://" + args.host + ":" + port + "/api/"

	data = { "task_id": [], "user_id": [], "info": [] }
	while remaining_data:
		address = prefix + "taskrun?limit=" + str(limit) + "&offset=" + str(offset)
		my_json = requests.get(address).json()
		offset = offset + limit
		if not my_json:
			logging.warning("Collection complete.")
			remaining_data = False
			return pd.DataFrame(data)
		for item in my_json:
			if item["project_id"] == project_id:
				if "info" in data:
					del data["info"]
					for key in item["info"].keys():
						data[key] = []
				data["task_id"].append(item["task_id"])
				data["user_id"].append(item["user_id"])
				my_info = item["info"]
				for key in item["info"]:
					data[key].append(item["info"][key])

def get_task_df(input_file):
	df = pd.read_csv(input_file, error_bad_lines=False,
		encoding='utf-8', na_filter=False, sep=',')
	return df

if __name__ == "__main__":
	logging.warning("Starting program")
	args = parse_arguments()
	project_id = get_project_id(args)
	logging.warning("Project ID is: {0}".format(project_id))
	df = get_task_run_df(args, project_id)
	idk = df
	idk_2 = get_task_df("meerkat/fat_head/pybossa/question_Starbucks_bank.csv")
	rows, _ = idk_2.shape
	idk_2.index = range(args.offset, rows + args.offset)
	idk_2["task_id"] = idk_2.index
	idk_3 = idk.join(idk_2, on="task_id", how="inner", lsuffix="left", rsuffix="right")
	slim_df = pd.DataFrame(idk_3, columns=["task_id", "user_id", "question", "city", "state",
		"zipcode", "not_in_us"])
	slim_df.to_csv("labeled_tasks.csv", sep="\t", index=False, quoting=csv.QUOTE_ALL)

	grouped = slim_df.groupby(["task_id"], as_index=True)

	redundancy = 2
	component_dataframes = []
	for name, group in grouped:
		original_count = len(group)
		dedup = group.drop_duplicates(subset=["city", "state", "zipcode", "not_in_us"])
		if len(dedup) == 1 and original_count == redundancy:
			component_dataframes.append(dedup)

	aligned_df = pd.concat(component_dataframes, axis=0)
	del aligned_df["user_id"]
	logging.warning("Found {0} unanimously labeled tasks".format(aligned_df.shape[0]))
	aligned_df.to_csv("aligned.csv", sep="\t", index=False)

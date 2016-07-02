import argparse
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

if __name__ == "__main__":
	logging.warning("Starting program")
	args = parse_arguments()
	project_id = get_project_id(args)
	logging.warning("Project ID is: {0}".format(project_id))
	df = get_task_run_df(args, project_id)
	logging.warning("Results dataframe is \n{0}".format(df))

"""This module builds pybossa projects for each top merchants using pbs commands"""

import os
import sys
import json
import inspect
import logging
import fileinput
import requests
import yaml

from ..tools import copy_file

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('build_pybossa_project')

def add_tasks(server, apikey, project_json_file, tasks_file):
	"""Add new tasks"""
	if not os.path.isfile(project_json_file):
		logger.critical("Cannot proceed, project json file not found")
		sys.exit()
	if not os.path.isfile(tasks_file):
		logger.critical("Cannot proceed, tasks file not found")
		sys.exit()
	os.system("pbs --server http://" + server + ":12000 --api-key " +
		apikey + " --project " + project_json_file + " add_tasks --tasks-file " +
		tasks_file)

def create_project_json_file(project_name, project_json_file):
	"""Create a json file for the new pybossa project"""
	project_json = {
		"name": project_name,
		"short_name": project_name,
		"description": project_name,
		"question": "geo"
	}
	with open(project_json_file, "w") as json_file:
		logger.info("Writing {0}".format(project_json_file))
		json.dump(project_json, json_file)

def format_json_with_callback(dictionary_file):
	"""Format the json dictionary to work with ajax callback"""
	# Prepend the json dictionary
	for line in fileinput.input(dictionary_file, inplace=True):
		if fileinput.isfirstline():
			print("callback(")
			print(line)
		else:
			print(line)

	# Append the json dictionary
	with open(dictionary_file, "a") as d_file:
		d_file.write(")")

def get_existing_projects(server, apikey):
	"""Get a list of existing pybossa projects"""
	port = "12000"
	address = "http://" + server + ":" + port + "/api/project?api_key=" + apikey
	my_json = requests.get(address).json()
	short_names = []
	for item in my_json:
		short_names.append(item["short_name"])
	return short_names

def get_top_merchant_names(base_dir):
	"""Get a list of top merchants that has dictionaries from agg data"""
	top_merchants = []
	merchants_with_paths = [obj[0] for obj in os.walk(base_dir)]
	merchants_without_paths = p"foo"]
	for merchant_path in merchants_with_paths:
		merchant = merchant_path[merchant_path.rfind("/") + 1:]
		if merchant not in ["", "pybossa_project"]:
			dictionary_exist = False
			for filename in os.listdir(merchant_path):
				if filename.endswith('.json'):
					dictionary_exist = True
					break
			if dictionary_exist:
				top_merchants.append(merchant)
			else:
				logger.error("Merchant {0} doesn't have dictionaries".format(merchant))
	return top_merchants

class Worker:
	"""Contains methods and data pertaining to the processing of pybossa project"""
	def __init__(self, config):
		"""Constructor"""
		self.config = config

	def main_process(self):
		"""Execute the main program"""
		base_dir = "meerkat/geomancer/merchants/"
		os.makedirs(base_dir, exist_ok=True)
		merchants_with_preconditions = get_top_merchant_names(base_dir)

		top_merchants = []
		for merchant in self.config["merchants"]:
			if merchant in merchants_with_preconditions:
				logger.info("Merchant {0} has dictionaries.".format(merchant))
				top_merchants.append(merchant)

		logger.info("Top merchants project to be processed are: {0}".format(top_merchants))

		server, apikey = self.config["server"], self.config["apikey"]
		existing_projects = get_existing_projects(server, apikey)
		bank_or_card = self.config["bank_or_card"]

		for merchant in top_merchants:
			merchant_dir = base_dir + merchant + "/pybossa_project/" + bank_or_card + "/"
			os.makedirs(merchant_dir, exist_ok=True)

			project_json_file = merchant_dir + "project.json"
			project_name = "Geomancer_" + bank_or_card + "_" + merchant
			# Create a new pybossa project
			if 'create_project' in self.config and self.config["create_project"]:
				if project_name in existing_projects:
					logger.warning("Project {0} already exists".format(project_name))
				else:
					create_project_json_file(project_name, project_json_file)
					os.system("pbs --server http://" + server + ":12000 --api-key " +
						apikey + " --project " + project_json_file + " create_project")
			elif project_name not in existing_projects:
				logger.error("Project {0} doesn't exist. Please first create it".format(project_name))
				return

			merchant_presenter = merchant_dir + "presenter_code.html"
			template_dir = "meerkat/geomancer/pybossa/template/"

			update_presenter = False
			# Update pybossa presenter
			if 'update_presenter' in self.config and self.config["update_presenter"]:
				presenter_file = template_dir + "presenter_code.html"
				copy_file(presenter_file, merchant_dir)
				replace_str_in_file(merchant_presenter, "Geomancer_project", "Geomancer_" + bank_or_card + "_" + merchant)
				update_presenter = True

			# Update pybossa dictionary
			if 'update_dictionary' in self.config and self.config["update_dictionary"]:
				dictionary_dst = "/var/www/html/dictionaries/" + merchant + "/"
				os.makedirs(dictionary_dst, exist_ok=True)
				copy_file(base_dir + merchant + "/geo.json", dictionary_dst)

				dictionary_file = dictionary_dst + "/geo.json"
				format_json_with_callback(dictionary_file)

				replace_str_in_file(merchant_presenter, "merchant_name", merchant)
				replace_str_in_file(merchant_presenter, "server_ip", server)
				logger.info("updated presenter with new dictionary")
				update_presenter = True

			if update_presenter: 
				long_description_file = template_dir + "long_description.md"
				results_file = template_dir + "results.html"
				tutorial_file = template_dir + "tutorial.html"

				os.system("pbs --server http://" + server + ":12000 --api-key " +
					apikey + " --project " + project_json_file + " update_project --task-presenter " +
					merchant_presenter + " --long-description " + long_description_file +
					" --results " + results_file + " --tutorial " + tutorial_file)

			# Add new labeling tasks
			if 'add_tasks' in self.config and self.config["add_tasks"]:
				tasks_file = merchant_dir + "tasks.csv"
				add_tasks(server, apikey, project_json_file, tasks_file)

def replace_str_in_file(file_name, old_str, new_str):
	"""Replace the occurrences of old string with new string in a file"""
	for line in fileinput.input(file_name, inplace=True):
		print(line.rstrip().replace(old_str, new_str))

if __name__ == "__main__":
	logger.critical("You cannot run this from the command line, aborting.")

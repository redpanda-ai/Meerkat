"""This module builds pybossa projects for each top merchants using pbs commands"""

import os
import sys
import json
import argparse
import inspect
import logging
import fileinput
import requests
import yaml

from ..tools import copy_file

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('build_pybossa_project')

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
	existing_merchants = [obj[0] for obj in os.walk(base_dir)]
	for merchant_path in existing_merchants:
		merchant = merchant_path[merchant_path.rfind("/") + 1:]
		if merchant not in ["", "pybossa_project"]:
			dictionary_exist = False
			for filename in os.listdir(merchant_path):
				if filename.endswith('.json'):
					dictionary_exist = True
					break
			if dictionary_exist:
				top_merchants.append(merchant)
	return top_merchants

def main_process(args):
	"""Execute the main programe"""
	base_dir = "meerkat/geomancer/merchants/"
	os.makedirs(base_dir, exist_ok=True)

	top_merchants = get_top_merchant_names(base_dir)

	if args.merchant != "":
		if args.merchant in top_merchants:
			top_merchants = [args.merchant]
		else:
			logger.error("Merchant {0} doesn't have dictionaries".format(args.merchant))
			return

	logger.info("Top merchants project to be processed are: {0}".format(top_merchants))

	server, apikey = args.server, args.apikey
	existing_projects = get_existing_projects(server, apikey)

	for merchant in top_merchants:
		merchant_dir = base_dir + merchant + "/pybossa_project/"
		os.makedirs(merchant_dir, exist_ok=True)

		project_json_file = merchant_dir + "project.json"
		project_name = "Geomancer_" + merchant

		# Create a new pybossa project
		if 'create_project' in args and args.create_project:
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
		if 'update_presenter' in args and args.update_presenter:
			presenter_file = template_dir + "presenter_code.html"
			copy_file(presenter_file, merchant_dir)
			replace_str_in_file(merchant_presenter, "Geomancer", "Geomancer_" + merchant)
			update_presenter = True

		# Update pybossa dictionary
		if 'update_dictionary' in args and args.update_dictionary:
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
		if 'add_tasks' in args and args.add_tasks:
			tasks_file = merchant_dir + "tasks.csv"
			os.system("pbs --server http://" + server + ":12000 --api-key " +
				apikey + " --project " + project_json_file + " add_tasks --tasks-file " +
				tasks_file)

def parse_arguments(args):
	"""Parse arguments from command line"""
	parser = argparse.ArgumentParser()

	parser.add_argument("--merchant", default="")

	parser.add_argument("--server", default="52.26.175.156")
	parser.add_argument("--apikey", default="b151d9e8-0b62-432c-aa3f-7f654ba0d983")

	parser.add_argument("--create_project", action="store_true")

	parser.add_argument("--update_presenter", action="store_true")

	parser.add_argument("--update_dictionary", action="store_true")

	parser.add_argument("--add_tasks", action="store_true")

	args = parser.parse_args(args)
	return args

def replace_str_in_file(file_name, old_str, new_str):
	"""Replace the occurrences of old string with new string in a file"""
	for line in fileinput.input(file_name, inplace=True):
		print(line.rstrip().replace(old_str, new_str))

if __name__ == "__main__":
	args = parse_arguments(sys.argv[1:])
	main_process(args)

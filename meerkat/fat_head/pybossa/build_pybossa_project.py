"""This module builds pybossa projects for each top merchants using pbs commands"""

import os
import sys
import json
import argparse
import inspect
import logging

from meerkat.various_tools import load_params

def parse_arguments(args):
	"""Parse arguments from command line"""
	parser = argparse.ArgumentParser()
	module_path = inspect.getmodule(inspect.stack()[1][0]).__file__
	default_project_dir = module_path[:module_path.rfind("/") + 1] + "projects/"
	default_dictionary_dir = "meerkat/fat_head/dictionaries/"

	parser.add_argument("--project_dir", default=default_project_dir)
	parser.add_argument("--dictionary_dir", default=default_dictionary_dir)
	parser.add_argument("--merchant", default="")

	parser.add_argument("--server", default="52.26.175.156")
	parser.add_argument("--apikey", default="b151d9e8-0b62-432c-aa3f-7f654ba0d983")

	parser.add_argument("--create_project", action="store_true")

	parser.add_argument("--update_project", action="store_true")
	parser.add_argument("--task_presenter", default="presenter_code.html")

	parser.add_argument("--add_tasks", action="store_true")
	parser.add_argument("--tasks_file", default="tasks1.csv")

	args = parser.parse_args(args)
	return args

def main_process():
	"""Execute the main programe"""
	log_format = "%(asctime)s %(levelname)s: %(message)s"
	logging.basicConfig(format=log_format, level=logging.INFO)

	args = parse_arguments(sys.argv[1:])

	project_dir = args.project_dir
	os.makedirs(project_dir, exist_ok=True)
	logging.info("Pybossa projects: {0}".format(project_dir))

	top_merchants = []
	dictionary_dir = args.dictionary_dir
	logging.info("Dictionaries from agg data: {0}".format(dictionary_dir))
	existing_dictionaries = [obj[0] for obj in os.walk(dictionary_dir)]
	for merchant in existing_dictionaries:
		merchant = merchant[merchant.rfind("/") + 1:]
		if merchant != "":
			top_merchants.append(merchant)
	if args.merchant != "":
		if args.merchant in top_merchants:
			top_merchants = [args.merchant]
		else:
			logging.error("Merchant {0} doesn't have dictionaries".format(args.merchant))
			return

	top_merchants_maps = {}
	for merchant in top_merchants:
		name = merchant.replace(" ", "_")
		for mark in '!"#$%&\'()*+,-./:;<=>?@[\]^`{|}~':
			name = name.replace(mark, '')
		top_merchants_maps[name] = merchant
	top_merchants = list(top_merchants_maps.keys())

	logging.info(top_merchants)
	logging.info("Formatting top merchant names:")
	logging.info(top_merchants_maps)

	server = args.server
	apikey = args.apikey

	for merchant in top_merchants:
		merchant_dir = project_dir + merchant + "/"
		os.makedirs(merchant_dir, exist_ok=True)

		project_json_file = merchant_dir + "project.json"

		if args.create_project:
			project_name = "Geomancer_" + merchant
			project_json = {
				"name": project_name,
				"short_name": project_name,
				"description": project_name,
				"question": "geo"
			}
			with open(project_json_file, "w") as json_file:
				logging.info("Writing {0}".format(project_json_file))
				json.dump(project_json, json_file)

			os.system("pbs --server http://" + server + ":12000 --api-key " +
				apikey + " --project " + project_json_file + " create_project")

	"""
		if args.update_project:
			presenter_file = project_dir + args.task_presenter
			long_description_file = project_dir + "long_description.md"
			results_file = project_dir + "results.html"
			tutorial_file = project_dir + "tutorial.html"
			tasks_file = project_dir + args.tasks_file
			os.system("pbs --server http://" + server + ":12000 --api-key " +
				apikey + " --project " + project_json + " update_project --task-presenter " +
				presenter_file + " --long-description " + long_description_file +
				" --results " + results_file + " --tutorial " + tutorial_file)

	if args.add_tasks:
		os.system("pbs --server http://" + server + ":12000 --api-key " +
			apikey + " --project " + project_json + " add_tasks --tasks-file " +
			tasks_file)
	"""
if __name__ == "__main__":
	main_process()

"""This module builds pybossa projects for each top merchants using pbs commands"""

import os
import sys
import json
import argparse
import inspect
import logging
import fileinput
import requests

from meerkat.various_tools import load_params
from meerkat.fat_head.tools import copy_file

def parse_arguments(args):
	"""Parse arguments from command line"""
	parser = argparse.ArgumentParser()
	module_path = inspect.getmodule(inspect.stack()[1][0]).__file__
	base_dir = module_path[:module_path.rfind("/") + 1]
	default_project_dir = base_dir + "projects/"
	default_dictionary_dir = "meerkat/fat_head/dictionaries/"

	parser.add_argument("--project_dir", default=default_project_dir)
	parser.add_argument("--dictionary_dir", default=default_dictionary_dir)
	parser.add_argument("--merchant", default="")

	parser.add_argument("--server", default="52.26.175.156")
	parser.add_argument("--apikey", default="b151d9e8-0b62-432c-aa3f-7f654ba0d983")

	parser.add_argument("--create_project", action="store_true")

	parser.add_argument("--update_presenter", action="store_true")
	parser.add_argument("--task_presenter", default="presenter_code.html")

	parser.add_argument("--update_dictionary", action="store_true")

	parser.add_argument("--add_tasks", action="store_true")
	parser.add_argument("--tasks_file", default="tasks.csv")

	args = parser.parse_args(args)
	return args

def format_merchant_names(top_merchants):
	"""Format merchant names"""
	top_merchants_maps = {}
	for merchant in top_merchants:
		name = merchant.replace(" ", "_")
		for mark in '!"#$%&\'()*+,-./:;<=>?@[\]^`{|}~':
			name = name.replace(mark, '')
		top_merchants_maps[name] = merchant
	top_merchants = list(top_merchants_maps.keys())

	logging.info("Formatting top merchant names:")
	logging.info(top_merchants_maps)
	return top_merchants, top_merchants_maps

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

	top_merchants, top_merchants_maps = format_merchant_names(top_merchants)
	logging.info(top_merchants)

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

		merchant_presenter = merchant_dir + args.task_presenter
		template_dir = "meerkat/fat_head/pybossa/template/"
		if args.update_presenter:
			presenter_file = template_dir + args.task_presenter
			copy_file(presenter_file, merchant_dir)
			for line in fileinput.input(merchant_presenter, inplace=True):
				print(line.rstrip().replace("Geomancer", "Geomancer_" + merchant))

		if args.update_dictionary:
			dictionary_dst = "/var/www/html/dictionaries/" + merchant + "/"
			os.makedirs(dictionary_dst, exist_ok=True)
			copy_file(dictionary_dir + top_merchants_maps[merchant] + "/geo.json", dictionary_dst)

			dictionary_file = dictionary_dst + "/geo.json"
			format_json_with_callback(dictionary_file)

			for line in fileinput.input(merchant_presenter, inplace=True):
				print(line.rstrip().replace("merchant_name", merchant))
			logging.info("updated presenter with new dictionary")

		long_description_file = template_dir + "long_description.md"
		results_file = template_dir + "results.html"
		tutorial_file = template_dir + "tutorial.html"
		if args.update_presenter or args.update_dictionary:
			logging.info("start pbs update")
			os.system("pbs --server http://" + server + ":12000 --api-key " +
				apikey + " --project " + project_json_file + " update_project --task-presenter " +
				merchant_presenter + " --long-description " + long_description_file +
				" --results " + results_file + " --tutorial " + tutorial_file)
			logging.info("finish pbs update")

		if args.add_tasks:
			tasks_file = merchant_dir + args.tasks_file
			os.system("pbs --server http://" + server + ":12000 --api-key " +
				apikey + " --project " + project_json_file + " add_tasks --tasks-file " +
				tasks_file)

if __name__ == "__main__":
	main_process()

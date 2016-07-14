#################### USAGE ##########################
"""
python3 -m meerkat.fat_head.pybossa.build_pybossa_project \
52.26.175.156 b151d9e8-0b62-432c-aa3f-7f654ba0d983 \
--create_project --update_project --add_tasks
"""
#####################################################

import os
import sys
import argparse

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("server")
	parser.add_argument("apikey")
	parser.add_argument("--create_project", action="store_true")
	parser.add_argument("--update_project", action="store_true")
	parser.add_argument("--add_tasks", action="store_true")
	args = parser.parse_args(args)
	return args

def main_process():
	args = parse_arguments(sys.argv[1:])
	server = args.server
	apikey = args.apikey
	base_dir = "meerkat/fat_head/pybossa/"
	project_json = base_dir + "geomencer_project.json"
	presenter_file = base_dir + "presenter_code.html"
	long_description_file = base_dir + "long_description.md"
	results_file = base_dir + "results.html"
	tutorial_file = base_dir + "tutorial.html"
	tasks_file = base_dir + "tasks.csv"

	if args.create_project:
		os.system("pbs --server http://" + server + ":12000 --api-key " +
			apikey + " --project " + project_json + " create_project")

	if args.update_project:
		os.system("pbs --server http://" + server + ":12000 --api-key " +
			apikey + " --project " + project_json + " update_project --task-presenter " +
			presenter_file + " --long-description " + long_description_file +
			" --results " + results_file + " --tutorial " + tutorial_file)

	if args.add_tasks:
		os.system("pbs --server http://" + server + ":12000 --api-key " +
			apikey + " --project " + project_json + " add_tasks --tasks-file " +
			tasks_file)

if __name__ == "__main__":
	main_process()

import argparse
import json
import pandas
import requests
import time


def get_project_id(args):
	port = "12000"
	prefix = "http://" + args.host + ":" + port + "/api/"
	address = prefix + "project"
	my_json = requests.get(address).json()
	for item in my_json:
		if item["short_name"] == args.short_name:
			return item["id"]
	raise Exception('Invalid short_name')

def get_task_run(args, project_id):
	offset, limit, port = 0, 100, "12000"
	remaining_data = True
	prefix = "http://" + args.host + ":" + port + "/api/"

	while remaining_data:
		address = prefix + "taskrun?limit=" + str(limit) + "&offset=" + str(offset)
		my_json = requests.get(address).json()
		offset = offset + limit
		if not my_json:
			print("My JSON is done.")
			remaining_data = False
			return;
		for item in my_json:
			if item["project_id"] == project_id:
				print("{0} {1} {2}".format(item["info"], item["task_id"], item["user_id"]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("host", help="IP or name of the server hosting the PyBossa project")
	parser.add_argument("short_name", help="Short name for the project")
	args = parser.parse_args()
	project_id = get_project_id(args)

	print("Project ID is: {0}".format(project_id))
	get_task_run(args, project_id)

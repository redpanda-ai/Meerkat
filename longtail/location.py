#!/usr/local/bin/python3
# pylint: disable=all

import json
from pprint import pprint
from longtail.clustering import cluster

def separate_results(result_list):
	"""Separate Results From Non Results"""

	hits = []
	non_hits = []

	for result in result_list:
		if result["pin.location"] == "":
			non_hits.append(result)
		else:
			hits.append(result)

	return hits, non_hits

def second_pass(result_list):

	full_results = []

	# Separate Results From Non Results
	hits, non_hits = separate_results(result_list)

	# Location List
	location_list = [json.loads(hit["pin.location"].replace("'", '"'))["coordinates"] for hit in hits]

	# Cluster Results
	cluster(location_list)

	return result_list

if __name__ == "__main__":
	""" Do Nothing"""
#!/usr/local/bin/python3
# pylint: disable=all

import json, sys
import pylab as pl
from pprint import pprint
from longtail.clustering import cluster
from longtail.scaled_polygon_test import scale_polygon
import matplotlib.pyplot as plt

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

def visualize(location_list, original_geoshapes, scaled_geoshapes):
	"""Visualize results of clustering and scaling in
	a useful way"""

	# Plot Points
	for point in location_list:
		print(point[1], point[0])
		pl.plot(point[0], point[1], 'o', markerfacecolor='k', markeredgecolor='k', markersize=5)

	pl.show()

def second_pass(result_list):

	full_results = []

	# Separate Results From Non Results
	hits, non_hits = separate_results(result_list)

	# Location List
	location_list = [json.loads(hit["pin.location"].replace("'", '"'))["coordinates"] for hit in hits]

	visualize(location_list, [], [])
	sys.exit()

	# Cluster Results
	original_geoshapes = cluster(location_list)

	# Scale Clusters
	scaled_geoshapes = [scale_polygon(geoshape, scale=2.0)[1] for geoshape in original_geoshapes]

	# Visualize
	visualize(location_list, original_geoshapes, scaled_geoshapes)

	return result_list

if __name__ == "__main__":
	""" Do Nothing"""
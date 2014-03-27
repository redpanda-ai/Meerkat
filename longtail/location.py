#!/usr/local/bin/python3
# pylint: disable=all

import json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Polygon
from pprint import pprint
from longtail.clustering import cluster
from longtail.scaled_polygon_test import scale_polygon

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

	# Meta
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	# Plot Points
	point_lat = [float(point[1]) for point in location_list]
	point_lon = [float(point[0]) for point in location_list]
	plt.axis([37, 38, -122.7, -121.5])
	plt.plot(point_lon, point_lat, 'ro', color='black', ms=2.0)

	# Plot Original Shapes
	for i in range(len(original_geoshapes)):
		original = Polygon(original_geoshapes[i], closed=True, fill=False, color='red')
		scaled = Polygon(scaled_geoshapes[i], closed=True, fill=False, color='blue')
		ax1.add_patch(original)
		ax1.add_patch(scaled)
	
	plt.show()

def second_pass(result_list):

	full_results = []

	# Separate Results From Non Results
	hits, non_hits = separate_results(result_list)

	# Location List
	location_list = [json.loads(hit["pin.location"].replace("'", '"'))["coordinates"] for hit in hits]

	# Cluster Results
	original_geoshapes = cluster(location_list)

	# Scale Clusters
	scaled_geoshapes = [scale_polygon(geoshape, scale=1.5)[1] for geoshape in original_geoshapes]

	# Visualize
	visualize(location_list, original_geoshapes, scaled_geoshapes)

	return result_list

if __name__ == "__main__":
	""" Do Nothing"""
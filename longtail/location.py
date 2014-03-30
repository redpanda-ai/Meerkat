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

def separate_geo(result_list):
	"""Separate Results From Non Results"""

	hits = []
	non_hits = []

	for result in result_list:
		if result["pin.location"] == "":
			non_hits.append(result)
		else:
			hits.append(result)

	return hits, non_hits

def visualize(arguments):
	"""Visualize results of clustering and scaling in
	a useful way"""

	print("HERE: ", arguments)

	# Meta
	location_list, original_geoshapes, scaled_geoshapes = arguments
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	# Plot Points
	point_lat = [float(point[1]) for point in location_list]
	point_lon = [float(point[0]) for point in location_list]
	plt.axis([-122.7, -121.5, 37, 38])
	plt.plot(point_lat, point_lon, 'ro', color='black', ms=2.0)

	# Plot Original Shapes
	for i in range(len(original_geoshapes)):
		o_geo = [point[::-1] for point in original_geoshapes[i]]
		s_geo = [point[::-1] for point in scaled_geoshapes[i]]
		original = Polygon(o_geo, closed=True, fill=False, color='red')
		scaled = Polygon(s_geo, closed=True, fill=False, color='blue')
		ax1.add_patch(original)
		ax1.add_patch(scaled)
	
	#plt.show()

def second_pass(result_list):

	full_results = []

	# Separate Results From Non Results
	hits, non_hits = separate_geo(result_list)

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
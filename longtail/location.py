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
from random import random

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

def visualize(location_list, original_geoshapes, scaled_geoshapes, user_id):
	"""Visualize results of clustering and scaling in
	a useful way"""

	# Meta
	ax = plt.gca()
	#fig = plt.figure()
	#ax1 = fig.add_subplot(111)

	# Plot Points
	point_lat = [float(point[1]) for point in location_list]
	point_lon = [float(point[0]) for point in location_list]
	plt.plot(point_lat, point_lon, 'ro', color='black', ms=2.0)

	# Plot Original Shapes
	for i in range(len(original_geoshapes)):
		o_geo = [point[::-1] for point in original_geoshapes[i]]
		s_geo = [point[::-1] for point in scaled_geoshapes[i]]
		original = Polygon(o_geo, closed=True, fill=False, color='red')
		scaled = Polygon(s_geo, closed=True, fill=False, color='blue')
		ax.add_patch(original)
		ax.add_patch(scaled)

	plt.savefig("user-" + user_id + ".png")

if __name__ == "__main__":
	""" Do Nothing"""
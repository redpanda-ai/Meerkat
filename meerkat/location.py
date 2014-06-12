#!/usr/local/bin/python3.3
# pylint: disable=all

"""This module aims to collect the functionality 
related to geolocation used throughout Meerkat.

@author: Matthew Sevrens
@author: J. Andrew Key
"""

import json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Polygon
from pprint import pprint
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

def scale_polygon(list_of_points, scale=2.0):
	"""This function accepts a list of points representing a polygon and scales
	them about its centroid."""
	#Create a matrix from the list of points, M
	M = np.matrix(list_of_points)
	#Sum axis 0 (the columns) to produce a 1 x n matrix (row vector)
	column_sums = M.sum(axis=0, dtype='float')
	#Grab the shape M, to learn how many points are in the list
	num_of_points, _ = M.shape
	#Divide the column_sums by the number of points to find the
	#average value for each dimension
	centroid_vector = column_sums / num_of_points
	#Create a matrix built of centroids, C, that is the same shape as M
	C = np.resize(centroid_vector, M.shape)
	#Subtract C from M to create a matrix of deltas, D,
	#from each point in M to each point in C
	D = M - C
	#Scale the matrix of deltas (D) by the scale provided and call it DS
	DS = D * scale
	#Add the Scaled Delta matrix (DS) to the Centroid Matrix (C) and call it S
	S = C + DS
	#Return the centroid vector, and a list of points representing
	#the scaled polygon
	return centroid_vector, S.tolist(), M, S

def plot_double_polygon(polygon_points, scaled_polygon_points, S
	, zoom_out_factor=2.5):
	"""This draws a simple plot to demonstrate scaling."""
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.add_patch(Polygon(polygon_points, closed=True, fill=False,
		color='red'))
	ax1.add_patch(Polygon(scaled_polygon_points, closed=True, fill=False,
		color='blue'))
	#Fetch the minimum and maximum dimension values for the scaled
	#polygon S and store them as 1x2 row vectors
	min_dimension_values = S.min(axis=0)
	max_dimension_values = S.max(axis=0)
	#Stack these vectors vertically to make a matrix of boundaries, B
	B = np.vstack((min_dimension_values, max_dimension_values))
	#Calculate the range between the highest and lowest values for
	#each dimension
	dimension_ranges = max_dimension_values - min_dimension_values
	#Create a column vector to represent zoom_out, and scale it by a factor
	zoom_out = np.matrix([[-1], [1]]) * zoom_out_factor
	#Multiply the zoom_out into the dimension_ranges to make a zoom
	#matrix, Z
	Z = zoom_out * dimension_ranges
	#Add the zoom matrix Z to the original boundaries B, to get your final
	#page boundaries, P
	P = B + Z

	x_boundaries = (P[0, 0], P[1, 0])
	y_boundaries = (P[0, 1], P[1, 1])
	ax1.set_xlim(x_boundaries)
	ax1.set_ylim(y_boundaries)
	plt.show()

def visualize(location_list, original_geoshapes, scaled_geoshapes, user_id):
	"""Visualize results of clustering and scaling in
	a useful way"""

	# Meta
	ax = plt.gca()

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

	plt.savefig("data/output/user_shapes/user-" + user_id + ".png")

if __name__ == "__main__":
	""" Do Nothing"""

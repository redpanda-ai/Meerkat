#!/usr/local/bin/python3.3
# pylint: disable=C0301

"""This module clusters a list of a users previously found
transaction locations. Provided at least 25 latitude / longitude
pairs, this module can group these points into separate categories
such as work or home by analyzing how close the points are to 
each other. After clustering, we find the outermost points that 
bound a users typical shopping area represented as geopolygons, 
scale these polygons inward or outward, and then generate a 
geopolygon query for ElasticSearch. This added context allows 
Meerkat to resolve ambiguous transactions that may not include a 
City or State to aid in indentification. 

@author: Matthew Sevrens
"""

import sys
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from meerkat.location import scale_polygon
from pprint import pprint

def cluster(location_list):
	"""Cluster Points"""

	# Parse to float
	location_list[:] = [[float(x[1]), float(x[0])] for x in location_list]
	X = location_list

	X = StandardScaler().fit_transform(X)
	db = DBSCAN(eps=0.08, min_samples=3).fit(X)

	# Find Shapes
	geoshapes = collect_clusters(X, db.labels_, location_list)

	# Plot Results
	plot_clustering(db, X)

	return geoshapes

def plot_clustering(model, normalized_points):
	"""Plot results of clustering"""

	core_samples = model.core_sample_indices_
	labels = model.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	unique_labels = set(labels)
	colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Noise throws off visualization
			continue
		class_members = [index[0] for index in np.argwhere(labels == k)]
		cluster_core_samples = [index for index in core_samples if labels[index] == k]
		for index in class_members:
			x = normalized_points[index]
			markersize = 5
			pl.plot(x[1], x[0], 'o', markerfacecolor=col, markeredgecolor='k', markersize=markersize)

	pl.show()

def collect_clusters(scaled_points, labels, location_list):
	"""Remap Normalized Clusters to location_list
	and return base geoshapes for further processing"""

	unique_labels = set(labels)
	clusters, locations = [], []

	for label in unique_labels:
		if label == -1:
			continue
		cluster, location = [], []
		for index, item in enumerate(labels):
			if item == label:
				cluster.append(scaled_points[index])
				location.append(location_list[index])
		clusters.append(cluster)
		locations.append(location)

	geoshape_list = convex_hull(clusters, locations)

	#The previous version of "convex_hull" converted coordinates from floating point pairs
	#to strings pairs.  If you need that functionality, use the
	#"convert_geoshapes_coordinates_to_strings" function provided below.

	return geoshape_list

def convex_hull(clusters, locations):

	"""Takes a normalized set of clusters and a list of the
	original lat lon points and returns an array of points
	that bound those clusters"""

	locations = np.array(locations)
	geoshapes, scaled_geoshapes = [], []

	for index, cluster in enumerate(clusters):

		lat_lon_points = np.array(locations[index])
		points = np.array(cluster)
		hull = ConvexHull(points)
		geoshape = []

		# Draw lines
		#for simplex in hull.simplices:
			#plt.plot(points[simplex, 1], points[simplex, 0], 'k-')

		# Get Lat Lon Vertices
		for vertex in hull.vertices:
			geoshape.append([lat_lon_points[vertex, 0], lat_lon_points[vertex, 1]])

		# Elastic Search requires closed polygon, repeat first point
		geoshape.append(geoshape[0])

		# Add to the list
		geoshapes.append(geoshape)

	return geoshapes

def convert_geoshapes_coordinates_to_strings(geoshape_list):
	"""Returns a copy of geoshape_list where each coordinate is formatted as a comma
	separated pair of string values. """
	new_geoshape_list = []
	for geoshape in geoshape_list:
		new_geoshape = []
		new_geoshape_list.append(new_geoshape)
		for coordinate in geoshape:
			new_coordinate = []
			new_geoshape.append(new_coordinate)
			for dimension in coordinate:
				new_coordinate.append(str(dimension))

	return new_geoshape_list

if __name__ == "__main__":
	""" Do some stuff."""
	print("This module is a library that contains useful functions; it should not be run from the console.")

#!/usr/bin/python

"""This is where we keep functions that are useful enough to call from
within multiple scripts."""

import csv
import re
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon

def load_dict_list(file_name, encoding='utf-8', delimiter="|"):
	"""Loads a dictionary of input from a file into a list."""
	input_file = open(file_name, encoding=encoding, errors='replace')
	dict_list = list(csv.DictReader(input_file, delimiter=delimiter,
		quoting=csv.QUOTE_NONE))
	input_file.close()
	return dict_list

def write_dict_list(dict_list, file_name, encoding="utf-8", delimiter="|"):
	""" Saves a lists of dicts with uniform keys to file """

	with open(file_name, 'w') as output_file:
		dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=dict_list[0].keys(), extrasaction='ignore')
		dict_w.writeheader()
		dict_w.writerows(dict_list)

def numeric_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [r"\[", r"\]", r"'", r"\{", r"\}", r'"', r"/", r"-"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	return re.sub(cleanse_pattern, "", original_string)

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

def string_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	original_string = original_string.replace("OR", "or")
	original_string = original_string.replace("AND", "and")
	bad_characters = [r"\[", r"\]", r"\{", r"\}", r'"', r"/", r"\\", r"\:",
		r"\(", r"\)", r"-", r"\+", r">", r"!", r"\*", r"\|\|", r"&&", r"~"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	with_spaces = re.sub(cleanse_pattern, " ", original_string)
	return ' '.join(with_spaces.split())

def synonyms(transaction):
	"""Replaces transactions tokens with manually
	mapped factual representations"""

	rep = {
		"wal-mart" : "Walmart",
		"samsclub" : "Sam's Club",
		"usps" : "US Post Office",
		"qps" : "",
		"q03" : "",
		"lowes" : "Lowe's",
		"wholefds" : "Whole Foods",
		"Shell Oil" : "Shell Gas",
		"wm supercenter" : "Walmart",
		"exxonmobil" : "exxonmobil exxon mobil",
		"mcdonalds" : "mcdonald's",
		"costco whse" : "costco",
		"franciscoca" : "francisco ca"
	}

	transaction = transaction.lower()
	rep = dict((re.escape(k), v) for k, v in rep.items())
	pattern = re.compile("|".join(rep.keys()))
	text = pattern.sub(lambda m: rep[re.escape(m.group(0))], transaction)
	
	return text



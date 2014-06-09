#!/usr/bin/python

"""This is where we keep functions that are useful enough to call from
within multiple scripts."""

import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.patches import Polygon

def load_dict_list(file_name, encoding='utf-8', delimiter="|"):
	input_file = open(file_name, encoding=encoding, errors='replace')
	dict_list = list(csv.DictReader(input_file, delimiter=delimiter, quoting=csv.QUOTE_NONE))
	input_file.close()
	return dict_list

def numeric_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [r"\[", r"\]", r"'", r"\{", r"\}", r'"', r"/", r"-"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	return re.sub(cleanse_pattern, "", original_string)

def plot_double_polygon(polygon_points, scaled_polygon_points, S, zoom_out_factor = 2.5):
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
	zoom_out = np.matrix([[-1],[1]]) * zoom_out_factor
	#Multiply the zoom_out into the dimension_ranges to make a zoom 
	#matrix, Z
	Z = zoom_out * dimension_ranges
	#Add the zoom matrix Z to the original boundaries B, to get your final
	#page boundaries, P
	P = B + Z

	x_boundaries = (P[0,0], P[1,0])
	y_boundaries = (P[0,1], P[1,1])
	ax1.set_xlim(x_boundaries)
	ax1.set_ylim(y_boundaries)
	plt.show()

def safely_remove_file(filename):
	print("Removing {0}".format(filename))
	try:
		os.remove(filename)
	except OSError:
		print("Unable to remove {0}".format(filename))
	print("File removed.")

def scale_polygon(list_of_points, scale=2.0):
	"""This function accepts a list of points representing a polygon and scales
	them about its centroid."""
	#Create a matrix from the list of points, M
	M = np.matrix(list_of_points)
	#Sum axis 0 (the columns) to produce a 1 x n matrix (row vector)
	column_sums = M.sum(axis=0, dtype='float')
	#Grab the shape M, to learn how many points are in the list
	num_of_points, _ = M.shape
	#Divide the column_sums by the number of points to find the average value for each dimension
	centroid_vector = column_sums / num_of_points
	#Create a matrix built of centroids, C, that is the same shape as M
	C = np.resize(centroid_vector, M.shape)
	#Subtract C from M to create a matrix of deltas, D, from each point in M to each point in C
	D = M - C
	#Scale the matrix of deltas (D) by the scale provided and call it DS
	DS = D * scale
	#Add the Scaled Delta matrix (DS) to the Centroid Matrix (C) and call it S
	S = C + DS
	#Return the centroid vector, and a list of points representing the scaled polygon
	return centroid_vector, S.tolist(), M, S

def string_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	original_string = original_string.replace("OR", "or")
	original_string = original_string.replace("AND", "and")
	bad_characters = [r"\[", r"\]", r"\{", r"\}", r'"', r"/", r"\\", r"\:", r"\(", r"\)", r"-", r"\+", r">", r"!", r"\*", r"\|\|", r"&&", r"~", r"[0-9]{8,}"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	with_spaces = re.sub(cleanse_pattern, " ", original_string)
	return ' '.join(with_spaces.split())

def split_csv(filehandler, delimiter=',', row_limit=10000, 
	output_name_template='output_%s.csv', output_path='.', keep_headers=True):
	"""
	Adapted from Jordi Rivero:
	https://gist.github.com/jrivero
	Splits a CSV file into multiple pieces.
	
	A quick bastardization of the Python CSV library.

	Arguments:
		`row_limit`: The number of rows you want in each output file. 10,000 by default.
		`output_name_template`: A %s-style template for the numbered output files.
		`output_path`: Where to stick the output files.
		`keep_headers`: Whether or not to print the headers in each output file.

	Example usage:
		>> from various_tools import split_csv;
		>> split_csv(open('/home/ben/input.csv', 'r'));
	
	"""
	reader = csv.reader(filehandler, delimiter=delimiter)
	#Start at piece one
	current_piece = 1
	current_out_path = os.path.join(
		 output_path,
		 output_name_template  % current_piece
	)
	#Create a list of file pieces
	file_list = [current_out_path]
	current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
	current_limit = row_limit
	if keep_headers:
		headers = reader.__next__()
		current_out_writer.writerow(headers)
	#Split the file into pieces
	for i, row in enumerate(reader):
		if i + 1 > current_limit:
			current_piece += 1
			current_limit = row_limit * current_piece
			current_out_path = os.path.join( output_path, output_name_template  % current_piece)
			file_list.append(current_out_path)
			current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
			if keep_headers:
				current_out_writer.writerow(headers)
		current_out_writer.writerow(row)
	#Return complete list of chunks
	return file_list


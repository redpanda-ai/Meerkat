import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Polygon
import pylab as pl

def draw_plot(polygon_points, scaled_polygon_points):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.add_patch(Polygon(polygon_points, closed=True, fill=False, color='red'))
	ax1.add_patch(Polygon(scaled_polygon_points, closed=True, fill=False, color='blue'))
	ax1.set_xlim((0,20))
	ax1.set_ylim((0,20))
	plt.show()
	#pl.scatter(lons,lats)
	#pl.show()
	#pass

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
	#Create a matrix built of centroids, C that is the same shape as M
	C = np.resize(centroid_vector, M.shape)
	#Subtract C from M to create a matrix of deltas, D, from each point in M to each point in C
	D = M - C
	#Scale the matrix of deltas (D) by the scale provided and call it DS
	DS = D * scale
	#Add the Scaled Delta matrix (DS) to the Centroid Matrix (C) and call it S
	S = C + DS
	#Return the centroid vector, and a list of points representing the scaled polygon
	return centroid_vector, S.tolist(), M, S

if __name__ == "__main__":
	"""Do this stuff."""
	# a 4 x 2 matrix
	scaling_factor = 2
	original_polygon_points =\
	[[10.0, 5.0], [5.0, 8.0], [7.0, 14], [13.0, 14.0], [15.0, 8.0]]
#		[[11.0, 11.0], [11.0, 13.0], [13.0, 13.0], [13.0, 11.0]]
	centroid, scaled_polygon_points, M, S =\
		scale_polygon(original_polygon_points,scale=scaling_factor)
	print("Original Polygon Points are:\n{0}".format(original_polygon_points))
	print("Scaling Factor is {0}".format(scaling_factor))
	print("Centroid is {0}".format(centroid))
	print("Scaled Polygon Points are:\n{0}".format(scaled_polygon_points))

	draw_plot(original_polygon_points, scaled_polygon_points)
#	draw_plot(M)
#	draw_plot(S)

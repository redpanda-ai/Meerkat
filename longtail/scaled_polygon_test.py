import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Polygon
import pylab as pl

def draw_plot(polygon_points, scaled_polygon_points, S, zoom_out_factor = 2.5):
	"""This draws a simple plot to demonstrate scaling."""
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.add_patch(Polygon(polygon_points, closed=True, fill=False,
		color='red'))
	ax1.add_patch(Polygon(scaled_polygon_points, closed=True, fill=False,
		color='blue'))
	#Fetch the minimum and maximum dimension values as 1x2 row vectors
	min_dimension_values = S.min(axis=0)
	max_dimension_values = S.max(axis=0)
	#Stack these vectors vertically to make a 2x2 matrix, of dimension
	#boundaries (B)
	B = np.vstack((min_dimension_values, max_dimension_values))
	#Find the difference between the highest and lowest values for 
	#each dimension (1x2 row vector)
	dimension_ranges = max_dimension_values - min_dimension_values
	#Create a 2x1 column vector for zoom_out, that scales according to 
	#zoom_out_factor 
	zoom_out = np.matrix([[-1],[1]]) * zoom_out_factor
	#Create a 2x2 matrix that applies zoom_out over dimension ranges
	Z = zoom_out * dimension_ranges
	#Add the zoom matrix Z, to the boundaries matrix B, for the final
	#matrix containing the page boundaries P
	P = B + Z 

	x_boundaries = (P[0,0], P[1,0])
	y_boundaries = (P[0,1], P[1,1])

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

if __name__ == "__main__":
	"""Do this stuff."""
	scaling_factor = 2

	#[[10.0, 5.0], [5.0, 8.0], [7.0, 14], [13.0, 14.0], [15.0, 8.0]]
	original_polygon_points =\
	[[-122.392586, 37.782428], [-122.434139, 37.725378], [-122.462813, 37.725407], [-122.48432, 37.742723], [-122.482605, 37.753909], [-122.476587, 37.784143], [-122.446137, 37.798541], [-122.419482, 37.807829], [-122.418104, 37.808003], [-122.413038, 37.807794], [-122.397797, 37.792259], [-122.392586, 37.782428]]
#	[[10.0 ,6.0], [7.0, 8.0], [6.0, 11], [8.0,14.0], [12.0, 14.0], [14.0, 11.0], [13.0, 8.0]]
	centroid, scaled_polygon_points, _, S =\
		scale_polygon(original_polygon_points,scale=scaling_factor)
	print("Original Polygon Points are:\n{0}".format(original_polygon_points))
	print("Scaling Factor is {0}".format(scaling_factor))
	print("Centroid is {0}".format(centroid))
	print("Scaled Polygon Points are:\n{0}".format(scaled_polygon_points))

	draw_plot(original_polygon_points, scaled_polygon_points, S, zoom_out_factor = 2)

#!/usr/local/bin/python3
# pylint: disable=C0301

"""This script clusters a list of latitude/ longitude pairs"""

import sys
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

def cluster(location_list):

    # Parse to float
    location_list[:] = [[float(x[1]), float(x[0])] for x in location_list]
    X = location_list

    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.2, min_samples=6).fit(X)

    # Find Shapes
    collect_clusters(X, db.labels_, location_list)

    core_samples = db.core_sample_indices_
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            markersize = 5
        class_members = [index[0] for index in np.argwhere(labels == k)]
        cluster_core_samples = [index for index in core_samples
                                if labels[index] == k]
        for index in class_members:
            x = X[index]
            if index in core_samples and k != -1:
                markersize = 5
            else:
                markersize = 5
            pl.plot(x[1], x[0], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=markersize)

    pl.show() 

def collect_clusters(scaled_points, labels, location_list):
    unique_labels = set(labels)
    clusters = []
    locations = []

    for label in unique_labels:
        
        if label == -1: 
            continue

        cluster = []
        location = []

        for index, item in enumerate(labels):
            if item == label:
                cluster.append(scaled_points[index])
                location.append(location_list[index])

        clusters.append(cluster)
        locations.append(location)

    convex_hull(clusters, locations)

def convex_hull(clusters, locations):

    """Takes a normalized set of clusters and a list of the
    original lat lon points and returns an array of points
    that bound those clusters"""

    locations = np.array(locations)
    geoshapes = []

    for index, cluster in enumerate(clusters):

        lat_lon_points = np.array(locations[index])
        points = np.array(cluster)
        hull = ConvexHull(points)
        geoshape = []

        # Draw lines
        for simplex in hull.simplices:
            #plt.plot(points[simplex, 1], points[simplex, 0], 'o', zorder=10, markersize=6, markerfacecolor=[0.0, 0.0, 0.0, 1.0])
            plt.plot(points[simplex, 1], points[simplex, 0], 'k-')

        print("\n")

        # Get Lat Lon Vertices
        for vertex in hull.vertices:
            geoshape.append([str(lat_lon_points[vertex, 0]), str(lat_lon_points[vertex, 1])])

        # Elastic Search requires closed polygon, repeat first point 
        geoshape.append(geoshape[0])

        # Add to the list
        geoshapes.append(geoshape)

    #print("GEOSHAPES FOR ELASTICSEARCH:")    

    for geoshape in geoshapes:
        print(geoshape, "\n")

    return geoshapes           

if __name__ == "__main__":
    """ Do Nothing"""
    
    some_points = [['-122.349802', '37.590282'], ['-122.014735', '37.325314'], ['-122.014735', '37.325314'], ['-122.014735', '37.325314'], ['-122.014735', '37.325314'], ['-122.014735', '37.325314'], ['-122.014735', '37.325314'], ['-122.048859', '37.35186'], ['-122.417173', '37.760139'], ['-122.025696', '37.362081'], ['-122.152738', '37.459119'], ['-122.409775', '37.78841'], ['-122.409775', '37.78841'], ['-121.717514', '37.195568'], ['-121.717514', '37.195568'], ['-122.078167', '37.423247'], ['-122.411071', '37.622767'], ['-122.435425', '37.761898'], ['-122.042418', '37.370249'], ['-122.161011', '37.446983'], ['-122.03181', '37.35219'], ['-122.03181', '37.35219'], ['-122.03181', '37.35219'], ['-122.214691', '37.486717'], ['-121.989189', '37.352032'], ['-122.416', '37.8084'], ['-122.138035', '37.421817'], ['-122.138035', '37.421817'], ['-122.138035', '37.421817'], ['-122.138035', '37.421817'], ['-122.42057', '37.805831'], ['-122.271077', '37.51615'], ['-122.404787', '37.783873'], ['-121.975761', '37.35294'], ['-122.267611', '37.52755'], ['-121.901581', '37.309868'], ['-122.25211', '37.519517'], ['-122.407037', '37.785455'], ['-122.028481', '37.364767'], ['-122.152847', '37.459499'], ['-122.152847', '37.459499'], ['-122.031555', '37.375767'], ['-122.15961', '37.462192'], ['-122.030396', '37.37711'], ['-122.252294', '37.520709'], ['-122.015946', '37.322723'], ['-121.889503', '37.335018'], ['-121.889503', '37.335018'], ['-122.16207', '37.444074'], ['-121.921659', '37.377617'], ['-121.921659', '37.377617'], ['-121.921659', '37.377617'], ['-122.162277', '37.444855'], ['-121.923615', '37.398479'], ['-121.99202', '37.404446'], ['-122.029732', '37.377106'], ['-122.033577', '37.368721'], ['-122.033577', '37.368721'], ['-122.250122', '37.498173'], ['-122.250122', '37.498173'], ['-122.419945', '37.63549'], ['-122.213097', '37.477648'], ['-121.9955', '37.3815'], ['-121.9955', '37.3815'], ['-122.025818', '37.36068'], ['-122.01487', '37.328091'], ['-122.139412', '37.433372'], ['-122.023674', '37.367039'], ['-122.013008', '37.324989'], ['-122.013008', '37.324989'], ['-122.03141', '37.352791'], ['-122.03141', '37.352791'], ['-122.03141', '37.352791'], ['-122.03141', '37.352791'], ['-122.03141', '37.352791'], ['-122.03141', '37.352791'], ['-122.402008', '37.791321'], ['-122.096527', '37.392521'], ['-122.159431', '37.461449']]
    cluster(some_points)
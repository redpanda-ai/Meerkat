import numpy as np
import re
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator
from meerkat.classification.bloom_filter.trie import location_split
from meerkat.various_tools import load_params

def match_store_number(city, description):
	json_file = "target_cities.json"
	maps = load_params(json_file)
	states = maps[city]
	count = 0
	result, last_state = None, None
	for state in states.keys():
		for store_number in states[state]:
			index = description.find(store_number)
			if index != -1:
				print("city {0} state {1} store_number {2} description {3}".format(city, state, store_number, description))
				result = state
				if result != last_state:
					count += 1
				last_state = result
				if count > 1:
					return None
	return result

class SystemZeroClassifier(BaseEstimator, ClassifierMixin):
	"""Trie classifier to classify city and state"""
	def __init__(self, demo_param='demo'):
		self.demo_param = demo_param
		self.sub_cities = load_params("meerkat/geomancer/data/sub_cities.json")
		self.cities_list = load_params("meerkat/geomancer/data/cities_list.json")["cities"]

	def fit(self, X, y):
		"""A reference implementation of a fitting function for a classifier."""
		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		# Store the classes seen durind fit
		self.classes_ = unique_labels(y)

		self.X_ = X
		self.y_ = y
		# Return the classifier
		return self

	def predict(self, X):
		""" A reference implementation of a prediction for a classifier."""
		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])
		# Input validation
		#X = check_array(X)

		y_predict = []
		for my_item in X:
			item = location_split(my_item)
			if item:
				key = (item[0] + "," + item[1]).upper()
				if key in self.sub_cities:
					y_predict.append(self.sub_cities[key])
				else:
					y_predict.append(item[0].upper() + ',' + item[1].upper())
			else:
				not_found = True
				for city in self.cities_list:
					index = my_item.find(city)
					if index != -1:
						state = match_store_number(city, my_item)
						if state is not None:
							y_predict.append(city + ',' + state)
							not_found = False
							break
				if not_found:
					y_predict.append('')
		return y_predict

#check_estimator(SystemZeroClassifier)

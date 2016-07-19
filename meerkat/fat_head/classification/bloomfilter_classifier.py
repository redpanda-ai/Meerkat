import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator
from meerkat.classification.bloom_filter.find_entities import location_split

class BloomfilterClassifier(BaseEstimator, ClassifierMixin):
	"""Bloomfilter classifier to classify city and state"""
	def __init__(self, demo_param='demo'):
		self.demo_param = demo_param

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
		for item in X.apply(location_split):
			if item:
				y_predict.append(item[0].upper() + ',' + item[1].upper())
			else:
				y_predict.append('')
		return y_predict

#check_estimator(BloomfilterClassifier)

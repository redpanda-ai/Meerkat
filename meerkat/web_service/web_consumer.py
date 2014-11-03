#!/usr/local/bin/python3.3

"""This module enriches transactions with additional
data found by Meerkat

Created on Nov 3, 2014
@author: Matthew Sevrens
"""

class Web_Consumer():
	"""Acts as a web service client to process and enrich
	transactions in real time"""

	def __init__(self, params, transactions, hyperparameters, cities):
		"""Constructor"""

		self.params = params
		self.hyperparameters = hyperparameters
		self.cities = cities

	def classify(self, transactions):
		"""Classify a set of transactions"""

		return {}

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	print("This module is a Class; it should not be run from the console.")
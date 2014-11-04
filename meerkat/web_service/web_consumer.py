#!/usr/local/bin/python3.3

"""This module enriches transactions with additional
data found by Meerkat

Created on Nov 3, 2014
@author: Matthew Sevrens
"""

from meerkat.binary_classifier.load import select_model

BANK_CLASSIFIER = select_model("bank")
CARD_CLASSIFIER = select_model("card")

class Web_Consumer():
	"""Acts as a web service client to process and enrich
	transactions in real time"""

	def __init__(self, params, hyperparameters, cities):
		"""Constructor"""

		self.params = params
		self.hyperparameters = hyperparameters
		self.cities = cities

	def classify(self, data):
		"""Classify a set of transactions"""

		physical, non_physical = [], []

		# Determine Whether to Search
		for trans in data["transaction_list"]:

			label = BANK_CLASSIFIER(trans["description"])

			if label == "1":
				physical.append(trans)
			else:
				non_physical.append(trans)

		for trans in physical:
			print(trans["description"])

		return {}

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	print("This module is a Class; it should not be run from the console.")
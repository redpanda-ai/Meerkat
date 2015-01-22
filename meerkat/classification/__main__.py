#!/usr/local/bin/python3.3
"""This file is run first when executing binary_classifier as a subpackage"""

#from meerkat.binary_classifier.bay import process_list
from meerkat.custom_exceptions import InvalidArguments

import sys, logging


def start(arguments):
	"""This function does everything."""
	try:
		input_file = open(arguments[1], encoding='utf-8')
		#trans_list = input_file.read().splitlines()
		#physical = process_list(trans_list)
		#print(physical)
		input_file.close()
	except FileNotFoundError:
		print(arguments[1], " not found, aborting.")
		logging.error(arguments[1] + " not found, aborting.")
		sys.exit()

if len(sys.argv) != 2:
	raise InvalidArguments(msg="Incorrect number of arguments", expr=None)

#MAIN PROGRAM
start(sys.argv)

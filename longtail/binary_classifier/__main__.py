#!/usr/local/bin/python3

from longtail.binary_classifier.bay import predict_if_physical_transaction, process_list

import sys, logging

if len(sys.argv) != 2:
	raise InvalidArguments(msg="Incorrect number of arguments", expr=None)
try:
	input_file = open(sys.argv[1], encoding='utf-8')
	trans_list = input_file.read().splitlines()
	physical = process_list(trans_list)
	print(physical)		
	input_file.close()
except FileNotFoundError:
	print(sys.argv[1], " not found, aborting.")
	logging.error(sys.argv[1] + " not found, aborting.")
	sys.exit()	
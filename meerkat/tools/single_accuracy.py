#!/usr/local/bin/python3.3

"""This utility loads a single classifier, runs it over the given data, 
compares the classifier's answers to the

Created in October, 2015
@author: Matthew Sevrens
@author: J. Andrew Key
@author: Joseph Altmaier
"""

#pylint:disable=anomalous-backslash-in-string,pointless-string-statement
#################### USAGE ##########################
"""
python3 -m meerkat.tools.single_accuracy -m <path_to_classifier> \ 
-d <path_to_classifier_output_map> -D <path_to_test_data_name_mapping> \
-f <path_to_test_data>
"""
#####################################################

from meerkat.classification.lua_bridge import get_cnn_by_path
import meerkat.accuracy
import argparse

def get_parser():
	""" Create the parser """	
	parser = argparse.ArgumentParser(description="Test a given machine learning model" 
		"against the given labeled data")
	parser.add_argument('--testfile', '-f', required=True,
		help="path to the test data")
	parser.add_argument('--model', '-m', required=True,
		help="path to the model under test")
	parser.add_argument('--dictionary', '-d', required=True,
		help="mapping of model output IDs to human readable names")
	parser.add_argument('--humandictionary', '-D', required=False,
		help="Optional mapping of human labels to the IDs output by your CNN. Default is None")
	parser.add_argument('--labelkey', '-l', required=False,
		default=meerkat.accuracy.default_label_key,
		help="Optional column containing the human label for a transaction."
		"Default is 'GOOD_DESCRIPTION'")
	parser.add_argument('--inputcolumn', '-i', required=False,
		default=meerkat.accuracy.default_doc_key,
		help="Optional column containing the data to be input to the CNN."
		"Default is 'DESCRIPTION_UNMASKED'")
	return parser

def run_from_command_line(args):
	"""Runs these commands if the module is invoked from the command line"""

	classifier = get_cnn_by_path(args.model, args.dictionary)
	meerkat.accuracy.print_results(meerkat.accuracy.CNN_accuracy(args.testfile, classifier, 
		model_dict=args.dictionary, human_dict=args.humandictionary, label_key=args.labelkey, doc_key=args.inputcolumn))

if __name__ == "__main__":
	run_from_command_line(get_parser().parse_args())

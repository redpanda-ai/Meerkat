#!/usr/local/bin/python3.3

"""This utility loads a single classifier, runs it over the given data, 
compares the classifier's answers to the

Created in October, 2015
@author: Matthew Sevrens
@author: J. Andrew Key
@author: Joseph Altmaier
"""

#################### USAGE ##########################

# python3 -m meerkat.tools.single_accuracy -m PATH_TO_CLASSIFIER -d PATH_TO_CLASSIFIER_OUTPUT_MAP -D PATH_TO_TEST_DATA_NAME_MAPPING -f PATH_TO_TEST_DATA

#####################################################

from meerkat.classification.lua_bridge import get_cnn_by_path
import meerkat.accuracy
import argparse

parser = argparse.ArgumentParser(description="Test a given machine learning model against the given labeled data")
parser.add_argument('--testfile', '-f', required=True, \
	help="path to the test data")
parser.add_argument('--model', '-m', required=True, \
	help="path to the model under test")
parser.add_argument('--dictionary', '-d', required=True, \
	help="mapping of model output IDs to human readable names")
parser.add_argument('--humandictionary', '-D', required=False, \
	help="Optional mapping of human labels to the IDs output by your CNN.  Default is None")
parser.add_argument('--labelkey', '-l', required=False, \
	default=meerkat.accuracy.default_label_key, \
	help="Optional column containing the human label for a transaction.  Default is 'GOOD_DESCRIPTION'")
parser.add_argument('--inputcolumn', '-i', required=False, \
	default=meerkat.accuracy.default_doc_key, \
	help="Optional column containing the data to be input to the CNN.  Default is 'DESCRIPTION_UNMASKED'")

def run_from_command_line(args):
	"""Runs these commands if the module is invoked from the command line"""

	classifier = get_cnn_by_path(args.model, args.dictionary)
	meerkat.accuracy.print_results(meerkat.accuracy.CNN_accuracy(args.testfile, classifier, args.dictionary, args.humandictionary, args.labelkey, args.inputcolumn))

if __name__ == "__main__":
	cmd_args = parser.parse_args()
	run_from_command_line(cmd_args)

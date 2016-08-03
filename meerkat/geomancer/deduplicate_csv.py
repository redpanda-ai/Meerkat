
#################### USAGE ##########################

# Use the -h flag from the command line to get help

#####################################################

import argparse
import csv
import sys
import logging
import pandas as pd
from .tools import deduplicate_csv

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file")
	parser.add_argument("--subset", default="")
	return parser.parse_args(args)

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	args = parse_arguments(sys.argv[1:])
	deduplicate_csv(args.input_file, args.subset)

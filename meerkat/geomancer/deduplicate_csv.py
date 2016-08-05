
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
	parser.add_argument("--inplace", action='store_true')
	return parser.parse_args(args)

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	args = parse_arguments(sys.argv[1:])
	read_csv_kwargs = { "error_bad_lines": False, "warn_bad_lines": True, "encoding": "utf-8",
		"quotechar" : '"', "na_filter" : False, "sep": "," }
	to_csv_kwargs = {"sep": ","}
	csv_kwargs = {
		"read": read_csv_kwargs,
		"to": to_csv_kwargs
	}
	deduplicate_csv(args.input_file, args.subset, args.inplace, **csv_kwargs)

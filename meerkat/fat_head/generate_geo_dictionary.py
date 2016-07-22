import argparse
import logging
import sys

from .tools import get_geo_dictionary

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file", help="csv containing city, state, and zip")
	return parser.parse_args(args)

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	args = parse_arguments(sys.argv[1:])
	logging.info(get_geo_dictionary(args.input_file))


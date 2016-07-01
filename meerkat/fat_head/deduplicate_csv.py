
#################### USAGE ##########################

# python3 -m meerkat.fat_head.deduplicate_csv Starbucks_bank.csv
# python3 -m meerkat.fat_head.deduplicate_csv Starbucks_bank.csv --subset DESCRIPTION_UNMASKED

#####################################################

import csv
import sys
import argparse
import logging
import pandas as pd

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file")
	parser.add_argument("--subset", default="")
	args = parser.parse_args(args)
	return args

def main_process():
	logging.basicConfig(level=logging.INFO)
	args = parse_arguments(sys.argv[1:])

	input_file = args.input_file
	subset = args.subset

	df = pd.read_csv(input_file, error_bad_lines=False,
		encoding='utf-8', na_filter=False, sep=',')
	if subset == "":
		unique_df = df.drop_duplicates(keep="first")
	else:
		unique_df = df.drop_duplicates(subset=subset, keep="first")

	logging.info("reduced {0} duplicate transactions".format(len(df) - len(unique_df)))
	output_file = 'deduplicated_' + input_file
	unique_df.to_csv(output_file, sep=',', index=False, quoting=csv.QUOTE_ALL)
	logging.info("csv files with unique {0} transactions saved to: {1}".format(len(unique_df), output_file))

if __name__ == "__main__":
	main_process()

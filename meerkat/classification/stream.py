import re
import sys
import logging

from .tools import (parse_arguments, cap_first_letter, pull_from_s3, slice_into_dataframes,
	convert_csv_to_torch_7_binaries, create_new_configuration_file, copy_file, execute_main_lua)

"""Streamlines the entire CNN training process."""

def main_stream():
	"""It all happens here"""
	args = parse_arguments()
	#1. Grab the input file from S3
	if args.input:
		logging.warning("Overriding default S3 location to {0}".format(args.input))
		my_regex = re.compile("^([^/]+)/(.*$)")
		if my_regex.search(args.input):
			results = my_regex.match(args.input)
			bucket, prefix = results.group(1), results.group(2)
			logging.warning("Valid S3 location: bucket {0}, prefix {1}".format(bucket, prefix))
		else:
			logging.critical("Invalid S3 location, aborting")
			sys.exit()
	else:
		#Download files from Card or Bank directory.
		logging.warning("Using default S3 location.")
		bucket = "yodleemisc"
		prefix = "hvudumala/Type_Subtype_finaldata/" + cap_first_letter(args.card_or_bank)

	my_filter, input_path = "csv", "./"
	input_file = pull_from_s3(bucket=bucket, prefix=prefix, my_filter=my_filter,
		input_path=input_path)
	#Temp
	sys.exit()
	#2.  Slice it into dataframes and make a mapping file.
	output_path = args.output_dir
	if output_path[-1:] != "/":
		output_path += "/"
	bank_or_card, debit_or_credit = args.card_or_bank, args.debit_or_credit
	train_poor, test_poor, num_of_classes = slice_into_dataframes(input_file=input_file,
		debit_or_credit=debit_or_credit, output_path=output_path, bank_or_card=bank_or_card)
	#3.  Use qlua to convert the files into training and testing sets.
	train_file = convert_csv_to_torch_7_binaries(train_poor)
	test_file = convert_csv_to_torch_7_binaries(test_poor)
	#4 Create a new configuration file based on the number of classes.
	create_new_configuration_file(num_of_classes, output_path, train_file, test_file)
	#5 Copy main.lua and data.lua to output directory.
	copy_file("meerkat/classification/lua/main.lua", output_path)
	copy_file("meerkat/classification/lua/data.lua", output_path)
	copy_file("meerkat/classification/lua/model.lua", output_path)
	copy_file("meerkat/classification/lua/train.lua", output_path)
	copy_file("meerkat/classification/lua/test.lua", output_path)
	#6 Excuete main.lua.
	execute_main_lua(output_path, "main.lua")

# The main program starts here if run from the command line.
if __name__ == "__main__":
	main_stream()


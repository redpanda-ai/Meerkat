import csv
import json
import logging
import pandas as pd
import sys

#USAGE:
#python3 -m meerkat.tools.step_1 3_year_bank_sample.txt 

def get_lookup_dicts(input_dataframe):
	# Normalize GOOD_DESCRIPTION to lowercase before making a histogram
	input_dataframe["GOOD_DESCRIPTION"] = input_dataframe.GOOD_DESCRIPTION.apply(
		lambda x: str(x).lower() if str(x).lower() != "nan" else "" )	
	# Get a histogram of the top 500 GOOD_DESCRIPTION classes; convert to dataframe
	histogram_dataframe = input_dataframe.GOOD_DESCRIPTION.value_counts().head(600).to_frame()
	# Save room for the null class
	offset = 1
	# Give each non-NULL class a number starting at 2
	histogram_dataframe['INDEX'] = list(range(offset, len(histogram_dataframe.index) + offset))
	# Convert the dataframe to a dict, but you only need the "INDEX" column
	classes_dict = histogram_dataframe.to_dict()["INDEX"]
	# Add a NULL class and set its number to 1
	#Convert all values to integers and all keys to lower-case
	#print(c)
	lookup, reverse_lookup = {}, {}
	#Build lookup dicts
	#bucket = 2
	#print(classes_dict)
	#sys.exit()
	for key in classes_dict.keys():
		key = key.lower()
		if key not in lookup:
			lookup[key] = int(classes_dict[key])
			#bucket += 1
		else:
			print("{0} key seen before".format(key))
	#Add a null class
	lookup[""] = 1

	for key in lookup.keys():
		new_key = lookup[key]
		reverse_lookup[lookup[key]] = key
	#Write lookup.json out to a file
	with open("lookup.json", 'w') as outfile:
		outfile.write(json.dumps(lookup, sort_keys=True, indent=4, separators=(',', ': ')))
	#Write reverse_lookup.json out to a file
	with open("reverse_lookup.json", 'w') as outfile:
		outfile.write(json.dumps(reverse_lookup, sort_keys=True, indent=4, separators=(',', ': ')))
	return lookup, reverse_lookup

def get_transformed_training_data(lookup, input_dataframe):
	#Pull out two columns
	sub_frame = input_dataframe.ix[:, ['GOOD_DESCRIPTION', 'DESCRIPTION_UNMASKED'] ]
	#sub_frame["DESCRIPTION_UNMASKED"] = sub_frame.DESCRIPTION_UNMASKED.apply(
	#	lambda x: str(x)[:])
	#Convert the good description to a class
	sub_frame["GOOD_DESCRIPTION"] = sub_frame.GOOD_DESCRIPTION.apply(
		lambda x: str(x).lower() if str(x).lower() != "nan" else "" )
	sub_frame["CLASS"] = sub_frame.GOOD_DESCRIPTION.apply(
		lambda x: lookup.get(x, "1"))
	#Pull out the two columns we want
	return sub_frame.ix[:, ['CLASS', 'GOOD_DESCRIPTION', 'DESCRIPTION_UNMASKED']]

def write_training_and_testing_csv(df):
	df.to_csv("train.csv", quoting=csv.QUOTE_ALL, chunksize=1000,
		columns=["CLASS", "GOOD_DESCRIPTION", "DESCRIPTION_UNMASKED"], header=False,
		index=False)

#Main function
if __name__ == "__main__":
	logging.warning("Building histogram")
	local_src_file = sys.argv[1]
	logging.warning("Analyzing {0}".format(local_src_file))
	initial_dataframe = pd.read_csv(local_src_file, sep='|', encoding="utf-8", error_bad_lines=False)

	lookup, reverse_lookup = get_lookup_dicts(initial_dataframe)
	df = get_transformed_training_data(lookup, initial_dataframe)
	write_training_and_testing_csv(df)


#!/usr/local/bin/python3
# pylint: disable=C0103

"""This script merges a new set of labeled
 transactions into our existing set."""

import csv, sys

file_a = open(sys.argv[1])
file_b = open("../data/labeledTrans.csv")
dict_a = csv.DictReader(file_a)
dict_b = csv.DictReader(file_b)
file_copy = []
field_names = dict_b.fieldnames

# Copy
for lRow in dict_b:
	file_copy.append(lRow)

# Merge
for row in dict_a:
	for lRow in file_copy:
		if (row['DESCRIPTION'] == lRow['DESCRIPTION']
			and lRow['PERSISTENTRECORDID'] == ""):
			lRow['PERSISTENTRECORDID'] = row['PERSISTENTRECORDID']
			break

# Write
output = open('output.csv', 'w')
csv_writer = csv.DictWriter(output, delimiter=',', fieldnames=field_names)
csv_writer.writeheader()
for row in file_copy:
	csv_writer.writerow(row)
output.close()


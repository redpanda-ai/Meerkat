#!/usr/local/bin/python3

"""This script merges a new set of labeled
 transactions into our existing set."""

import csv, sys

FILE_A = open(sys.argv[1])
FILE_B = open("../labeledTrans.csv")
DICT_A = csv.DictReader(FILE_A)
DICT_B = csv.DictReader(FILE_B)
FILE_COPY = []
FIELD_NAMES = DICT_B.fieldnames

# Copy
for lRow in DICT_B:
	FILE_COPY.append(lRow)

# Merge
for row in DICT_A:
	for lRow in FILE_COPY:
		if (row['DESCRIPTION'] == lRow['DESCRIPTION']
			and lRow['PERSISTENTRECORDID'] == ""):
			lRow['PERSISTENTRECORDID'] = row['PERSISTENTRECORDID']
			break

# Write
OUTPUT = open('output.csv', 'w')
CSV_WRITER = csv.DictWriter(OUTPUT, delimiter=',', fieldnames=FIELD_NAMES)
CSV_WRITER.writeheader()
for row in FILE_COPY:
	CSV_WRITER.writerow(row)
OUTPUT.close()


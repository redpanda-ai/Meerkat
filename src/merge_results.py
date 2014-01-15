#!/usr/local/bin/python3

"""This script merges a new set of labeled
 transactions into our existing set."""

import csv, sys

FILE_A = open(sys.argv[1])
FILE_B = open("../labeledTrans.csv")
DICT_A = csv.DictReader(FILE_A)
DICT_B = csv.DictReader(FILE_B)
fileCopy = []
fieldnames = DICT_B.fieldnames

# Copy
for lRow in DICT_B: 
	fileCopy.append(lRow)

# Merge
for row in DICT_A:
	for lRow in fileCopy:
		if row['DESCRIPTION'] == lRow['DESCRIPTION'] and lRow['PERSISTENTRECORDID'] == "":
			lRow['PERSISTENTRECORDID'] = row['PERSISTENTRECORDID']	
			break

# Write
output = open('output.csv', 'w')
csvwriter = csv.DictWriter(output, delimiter=',', fieldnames=fieldnames)
csvwriter.writeheader()
for row in fileCopy:
	csvwriter.writerow(row)
output.close()


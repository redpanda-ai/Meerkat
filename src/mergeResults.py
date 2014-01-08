#!/usr/local/bin/python3

"""This script merges a new set of labeled transactions into our 
	existing set."""

import csv, sys

fileA = open(sys.argv[1])
fileB = open("../labeledTrans.csv")
dictA = csv.DictReader(fileA)
dictB = csv.DictReader(fileB)
fileCopy = []
fieldnames = dictB.fieldnames

# Copy
for lRow in dictB: 
	fileCopy.append(lRow)

# Merge
for row in dictA:
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


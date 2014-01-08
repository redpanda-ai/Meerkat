#!/usr/local/bin/python3

"""This script tests the current accuracy of our labeling tool"""

import csv, sys, math

fileA = open(sys.argv[1])
fileB = open("../labeledTrans.csv")
dictA = csv.DictReader(fileA)
dictB = csv.DictReader(fileB)
fileCopy = []
total = 0
correct = 0

# Copy
for lRow in dictB: 
	fileCopy.append(lRow)

# Merge
for row in dictA:
	total += 1
	for lRow in fileCopy:
		if row['DESCRIPTION'] == lRow['DESCRIPTION']:
			if row['PERSISTENTRECORDID'] == lRow['PERSISTENTRECORDID']:
				correct += 1
			break

print(str(math.ceil(correct/total * 100)) + "%")
		
#!/usr/local/bin/python3

"""This script tests the current accuracy of our labeling tool"""

import csv, sys, math

fileA = open(sys.argv[1])
fileB = open("../labeledTrans.csv")
dictA = csv.DictReader(fileA)
dictB = csv.DictReader(fileB)
fileCopy = []
total = 0
notFound = 0
correct = 0

# Copy
for lRow in dictB: 
	fileCopy.append(lRow)

# Count
numLabeled = len(fileCopy)
for row in dictA:
	total += 1
	for index, lRow in enumerate(fileCopy):
		if row['DESCRIPTION'] == lRow['DESCRIPTION']:
			if row['PERSISTENTRECORDID'] == lRow['PERSISTENTRECORDID']:
				correct += 1
			break
		if index + 1 == numLabeled:
			notFound += 1

print("Accuracy = " + str(math.ceil(correct/(total-notFound) * 100)) + "%")
print("Not Found = " + str(math.ceil(notFound/total * 100)) + "%")
		

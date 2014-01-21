#!/usr/local/bin/python3
# pylint: disable=C0103

"""This script tests the current accuracy of our labeling tool"""

import csv, sys, math

file_a = open(sys.argv[1])
file_b = open("../data/labeledTrans.csv")
dict_a = csv.DictReader(file_a)
dict_b = csv.DictReader(file_b)
file_copy = []
total = 0
not_found = 0
correct = 0

# Copy
for lRow in dict_b:
	file_copy.append(lRow)

# Count
num_labeled = len(file_copy)
for row in dict_a:
	total += 1
	for index, lRow in enumerate(file_copy):
		if row['DESCRIPTION'] == lRow['DESCRIPTION']:
			if row['PERSISTENTRECORDID'] == lRow['PERSISTENTRECORDID']:
				correct += 1
			break
		if index + 1 == num_labeled:
			not_found += 1

print("Accuracy = " + str(math.ceil(correct/(total-not_found) * 100)) + "%")
print("Not Found = " + str(math.ceil(not_found/total * 100)) + "%")

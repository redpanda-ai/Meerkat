#!/usr/local/bin/python3
# pylint: disable=C0103

"""This script tests the current accuracy of our labeling tool"""

import csv, sys, math

machine_labeled = open(sys.argv[1])
human_labeled = open("../data/labeledTrans.csv")
dict_ML = csv.DictReader(machine_labeled)
dict_HL = csv.DictReader(human_labeled)
file_copy = []
total = 0
not_found = 0
correct = 0

# Copy
for hlRow in dict_HL:
	file_copy.append(hlRow)

# Count
num_labeled = len(file_copy)
for mlRow in dict_ML:
	total += 1
	for index, hlRow in enumerate(file_copy):
		if mlRow['DESCRIPTION'] == hlRow['DESCRIPTION']:
			if mlRow['PERSISTENTRECORDID'] == hlRow['PERSISTENTRECORDID']:
				correct += 1
			else: print("MISLABELED", hlRow['PERSISTENTRECORDID'], hlRow['DESCRIPTION'])	
			break
		if index + 1 == num_labeled:
			print("NEEDS MANUAL VERIFICATION", mlRow['DESCRIPTION'])
			not_found += 1

print("Accuracy = " + str(math.ceil(correct/(total-not_found) * 100)) + "%")
print("Not Found = " + str(math.ceil(not_found/total * 100)) + "%")

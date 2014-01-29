#!/usr/local/bin/python3
# pylint: disable=C0103
# pylint: disable=C0301

"""This script tests the current accuracy of our labeling tool"""

import csv, sys, math

def test_accuracy(file_path):
	"""Docstring to be determined."""
	human_labeled = open("data/verifiedLabeledTrans.csv") if __name__ == "__main__" else file_path
	machine_labeled = open(sys.argv[1])
	dict_ML = csv.DictReader(machine_labeled)
	dict_HL = csv.DictReader(human_labeled)

	file_copy = []
	needs_hand_labeling = []
	non_physical = []
	mislabeled = []
	unlabeled = []

	total = 0
	correct = 0
	found_in_verified = 0

	# Copy
	for hlRow in dict_HL:
		file_copy.append(hlRow)

	# Count
	num_labeled = len(file_copy)
	for mlRow in dict_ML:
		total += 1
		for index, hlRow in enumerate(file_copy):
			if mlRow['DESCRIPTION'] == hlRow['DESCRIPTION']:
				found_in_verified += 1
				if mlRow['PERSISTENTRECORDID'] == hlRow['PERSISTENTRECORDID']:
					# Transaction was correctly labeled
					correct += 1
					break
				elif hlRow['IS_PHYSICAL_TRANSACTION'] == '0':
					# Transaction is non physical
					non_physical.append(mlRow['DESCRIPTION'])
					break
				elif hlRow['PERSISTENTRECORDID'] == "":
					# Transaction is not yet labeled
					needs_hand_labeling.append(mlRow['DESCRIPTION'])
					break
				elif mlRow['PERSISTENTRECORDID'] == "":
					# Our confidence was not high enough to label
					unlabeled.append(mlRow['DESCRIPTION'])
					break
				else:
					# Transaction is mislabeled
					mislabeled.append(hlRow['DESCRIPTION'] + " - " + hlRow['PERSISTENTRECORDID'])
					break
			if index + 1 == num_labeled:
				needs_hand_labeling.append(mlRow['DESCRIPTION'])

	print("STATS:")
	print("Accuracy = " + str(math.ceil(correct/(total-len(needs_hand_labeling)) * 100)) + "%")
	print("Not Found in hand verified list = " + str(math.ceil(len(needs_hand_labeling)/total * 100)) + "%")
	print("Incorrect binary classification = " + str(math.ceil(len(non_physical)/found_in_verified * 100)) + "%", '\n')
	print("NEEDS HAND LABELING:", '\n'.join(needs_hand_labeling), sep="\n")
	print("UNLABELED:", '\n'.join(unlabeled), sep="\n")
	print("", "MISLABELED:", '\n'.join(mislabeled), sep="\n")
	print("", "NON_PHYSICAL:", '\n'.join(non_physical), sep="\n")

if __name__ == "__main__":
	test_accuracy("data/verifiedLabeledTrans.csv")
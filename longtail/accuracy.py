#!/usr/local/bin/python3
# pylint: disable=C0103
# pylint: disable=C0301

"""This script tests the current accuracy of our labeling tool"""

import csv, sys, math, logging

def test_accuracy(file_path=None, non_physical=None, result_list=None):
	"""Takes file by default but can accept result
	queue/ non_physical list. Attempts to provide various
	accuracy tests"""

	if result_list is not None:
		machine_labeled = result_list
	elif file_path is not None:
		ML_file = open(file_path)
		machine_labeled = list(csv.DictReader(ML_file))
	else:
		logging.warning("Nothing provided to perform accuracy tests on")
		
	HL_file = open("data/misc/verifiedLabeledTrans.csv")
	human_labeled = list(csv.DictReader(HL_file))

	needs_hand_labeling = []
	non_physical = []
	mislabeled = []
	unlabeled = []

	total = len(machine_labeled)
	correct = []
	num_found = 0

	# Test Recall / Precision
	for mlRow in machine_labeled:

		# Our confidence was not high enough to label
		if mlRow['PERSISTENTRECORDID'] == "":
			unlabeled.append(mlRow['DESCRIPTION'])
			continue

		# Verify against human labeled
		for index, hlRow in enumerate(human_labeled):
			if mlRow['DESCRIPTION'] == hlRow['DESCRIPTION']:
				if hlRow['PERSISTENTRECORDID'] == "":
					# Transaction is not yet labeled
					needs_hand_labeling.append(mlRow['DESCRIPTION'])
					break
				elif mlRow['PERSISTENTRECORDID'] == hlRow['PERSISTENTRECORDID']:
					# Transaction was correctly labeled
					correct.append(hlRow['DESCRIPTION'] + " (ACTUAL:" + hlRow['PERSISTENTRECORDID'] + ")")
					break
				elif hlRow['IS_PHYSICAL_TRANSACTION'] == '0':
					# Transaction is non physical
					non_physical.append(mlRow['DESCRIPTION'])
					break
				else:
					# Transaction is mislabeled
					mislabeled.append(hlRow['DESCRIPTION'] + " (ACTUAL:" + hlRow['PERSISTENTRECORDID'] + ")")
					break
			elif index + 1 == len(human_labeled):
				needs_hand_labeling.append(mlRow['DESCRIPTION'])

	# Test Binary
	for i, item in enumerate(unlabeled):
		for index, hlRow in enumerate(human_labeled):
			if item == hlRow['DESCRIPTION']:
				if hlRow['IS_PHYSICAL_TRANSACTION'] == '0':
					# Transaction is non physical
					non_physical.append(item)
					break


	num_labeled = total - len(unlabeled)
	num_verified = num_labeled - len(needs_hand_labeling)

	results = {}
	results['total'] = total
	results['correct'] = correct
	results['needs_hand_labeling'] = needs_hand_labeling
	results['non_physical'] = non_physical
	results['unlabeled'] = unlabeled
	results['mislabeled'] = mislabeled
	results['recall'] = math.ceil((num_labeled / total) * 100)
	results['precision'] = math.ceil((len(correct) / num_verified) * 100)
	results['incorrect_binary'] = math.ceil((len(non_physical) / total) * 100)

	return results

def print_results(results):
	
	#print("", "CORRECT:", '\n'.join(results['correct']), sep="\n")	
	print("\n", "STATS:")
	print("Recall = " + str(results['recall']) + "%")
	print("Precision = " + str(results['precision']) + "%")
	print("Binary Classifier Accuracy = " + str(100 - results['incorrect_binary']) + "%")
	print("", "MISLABELED:", '\n'.join(results['mislabeled']), sep="\n")	

if __name__ == "__main__":

	file_path = sys.argv[1] if len(sys.argv) > 1 else "data/output/longtailLabeled.csv"
	print_results(test_accuracy(file_path))

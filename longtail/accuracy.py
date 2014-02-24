#!/usr/local/bin/python3
# pylint: disable=C0103
# pylint: disable=C0301

"""This script tests the current accuracy of our labeling tool"""

import csv, sys, math, logging, datetime, os

def test_accuracy(file_path=None, non_physical_trans=[], result_list=[]):
	"""Takes file by default but can accept result
	queue/ non_physical list. Attempts to provide various
	accuracy tests"""

	if len(result_list) > 0:
		machine_labeled = result_list
	elif file_path is not None and os.path.isfile(file_path):
		ML_file = open(file_path)
		machine_labeled = list(csv.DictReader(ML_file))
	else:
		logging.warning("Not enough information provided to perform accuracy tests on")
		return

	HL_file = open("data/misc/verifiedLabeledTrans.csv")
	human_labeled = list(csv.DictReader(HL_file))
	HL_file.close()

	# Ensure there is something to process
	total = len(machine_labeled)
	total_processed = len(machine_labeled) + len(non_physical_trans)

	if total == 0 or total_processed == 0:
		logging.warning("Nothing provided to perform accuracy tests on")
		return

	non_physical_trans = non_physical_trans or []
	needs_hand_labeling = []
	non_physical = []
	mislabeled = []
	unlabeled = []
	correct = []

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
	for item in unlabeled:
		for index, hlRow in enumerate(human_labeled):
			if item == hlRow['DESCRIPTION']:
				if hlRow['IS_PHYSICAL_TRANSACTION'] == '0':
					# Transaction is non physical
					non_physical.append(item)
					break

	# Collect results into dict for easier access
	num_labeled = total - len(unlabeled)
	num_verified = num_labeled - len(needs_hand_labeling)
	num_verified = num_verified if num_verified > 0 else 1
	num_correct = len(correct)

	results = {}
	results['total_processed'] = total_processed
	results['total_physical'] = math.ceil((len(machine_labeled) / results['total_processed']) * 100)
	results['total_non_physical'] = math.ceil((len(non_physical_trans) / results['total_processed']) * 100)
	results['correct'] = correct
	results['needs_hand_labeling'] = needs_hand_labeling
	results['non_physical'] = non_physical
	results['unlabeled'] = unlabeled
	results['num_verified'] = num_verified
	results['mislabeled'] = mislabeled
	results['total_recall'] = math.ceil((num_labeled / results['total_processed']) * 100)
	results['total_recall_non_physical'] = math.ceil((num_labeled / total) * 100)
	results['precision'] = math.ceil((num_correct / num_verified) * 100)
	results['binary_accuracy'] = 100 - math.ceil((len(non_physical) / total) * 100)

	return results

def speed_tests(start_time, accuracy_results):
	"""Run a number of tests related to speed"""

	time_delta = datetime.datetime.now() - start_time
	time_per_transaction = time_delta.seconds / accuracy_results['total_processed']
	transactions_per_minute = (accuracy_results['total_processed'] / time_delta.seconds) * 60

	print("")
	print("SPEED TESTS:")
	print("Total Time Taken: " + str(time_delta))
	print("Time Per Transaction: " + str(time_per_transaction) + " seconds")
	print("Transactions Per Minute: " + str(transactions_per_minute))

	return {'time_delta':time_delta,
			'time_per_transaction': time_per_transaction,
			'transactions_per_minute':transactions_per_minute}

def print_results(results):
	"""Provide useful readable output"""

	if results is None:
		return

	print("")
	print("STATS:")
	print("Total Transactions Processed = " + str(results['total_processed']))
	print("Total Labeled Physical = " + str(results['total_physical']) + "%")
	print("Total Labeled Non Physical = " + str(results['total_non_physical']) + "%")
	print("Binary Classifier Accuracy = " + str(results['binary_accuracy']) + "%", "\n")
	print("Recall of all transactions = " + str(results['total_recall']) + "%")
	print("Recall of transactions labeled physical = " + str(results['total_recall_non_physical']) + "%")
	print("Number of transactions verified = " + str(results['num_verified']))
	print("Precision = " + str(results['precision']) + "%")
	print("", "MISLABELED:", '\n'.join(results['mislabeled']), sep="\n")

if __name__ == "__main__":

	output_path = sys.argv[1] if len(sys.argv) > 1 else "data/output/longtailLabeled.csv"
	print_results(test_accuracy(file_path=output_path))

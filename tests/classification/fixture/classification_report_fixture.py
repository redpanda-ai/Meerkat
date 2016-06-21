"""Fixtures for test_classification_report"""

BASE_DIR = "tests/classification/fixture/"

def get_csv():
	"""Return a csv file"""
	return BASE_DIR + "classification_report_sample.csv"

def get_confusion_matrix(case_type):
	"""Return a confusion matrix"""
	if case_type == "invalid_cf":
		return BASE_DIR + "invalid_confusion_matrix.csv"
	elif case_type == "valid_cf":
		return BASE_DIR + "valid_confusion_matrix.csv"

def get_label_map():
	"""Return a label map"""
	return {
		"1": "Bank Adjustment - Adjustment",
		"2": "Deposits & Credits - Rewards",
		"3": "Other Deposits - Credit"
	}

"""Fixtures for test_classification_report"""

def get_csv():
	"""Return a csv file"""
	return "tests/classification/fixture/classification_report_sample.csv"

def get_confusion_matrix(case_type):
	"""Return a confusion matrix"""
	if case_type == "invalid_cf":
		return "tests/classification/fixture/invalid_confusion_matrix.csv"

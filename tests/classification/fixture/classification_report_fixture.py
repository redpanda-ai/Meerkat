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

def get_machine_result():
	"""Return predicted result"""
	return [
		{
			'PROPOSED_SUBTYPE': 'Abc',
			'PREDICTED_CLASS': 'Abc',
			'PREDICTED_INDEX': '1',
			'DESCRIPTION_UNMASKED': 'NEWTON GRV BST NEWTON GROVE NC',
			'ACTUAL_INDEX': '1'
		},
		{
			'PROPOSED_SUBTYPE': 'Xyz',
			'PREDICTED_CLASS': 'Abc',
			'PREDICTED_INDEX': '1',
			'DESCRIPTION_UNMASKED': 'XXXX YYY ZZZZ',
			'ACTUAL_INDEX': '2'
		}
	]

def get_compare_label_result_non_fast():
	"""Retrun result of compare_label of non_fast mode"""
	mislabeled = [['XXXX YYY ZZZZ', 'Xyz', 'Abc']]
	correct = [['NEWTON GRV BST NEWTON GROVE NC', 'Abc']]
	conf_mat = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
	return mislabeled, correct, conf_mat

def get_compare_label_result_fast():
	"""Retrun result of compare_label of fast mode"""
	mislabeled = []
	correct = []
	conf_mat = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
	return mislabeled, correct, conf_mat

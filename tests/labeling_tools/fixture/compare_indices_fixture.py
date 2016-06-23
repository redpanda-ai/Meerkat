"""Fixtures for test_compare_indices"""

def get_elasticsearch_result():
	"""Return an elasticsearch result"""
	return {
		"non_hits": {
			"hits": {
				"total": 0
			}
		},
		"has_hits": {
			"hits": {
				"total": 2,
				"hits": [
					{"_source": "result_0", "_score": 2.0},
					{"_source": "result_1", "_score": 1.0}
				]
			}
		}
	}

def get_transaction():
	"""Return a transaction"""
	return {
		"CITY": "Scarsdale",
		"STATE": "NY",
		"LATITUDE": "234",
		"LONGITUDE": "123",
		"PHYSICAL_MERCHANT": "7 Eleven",
		"STORE_NUMBER": "00234",
		"STREET": "1212 MAY ST",
		"ZIP_CODE": "95048",
		"TRANSACTION_ID": 5024853,
		"TXN_TYPE": "Purchase",
		"TXN_SUB_TYPE": "Purchase",
		"WEBSITE": "",
		"PHONE_NUMBER": "",
		"COUNTRY": "US"
	}

def get_cleaned_transaction():
	"""Return a cleaned transaction"""
	return {
		"CITY": "",
		"STATE": "",
		"LATITUDE": "",
		"LONGITUDE": "",
		"PHYSICAL_MERCHANT": "",
		"STORE_NUMBER": "",
		"STREET": "",
		"ZIP_CODE": "",
		"TRANSACTION_ID": 5024853,
		"TXN_TYPE": "Purchase",
		"TXN_SUB_TYPE": "Purchase",
		"WEBSITE": "",
		"PHONE_NUMBER": "",
		"COUNTRY": "US"
	}

def get_args():
	"""Return different arguments"""
	return {
		"not_enough": ["arg_0", "arg_1"],
		"no_json": ["arg_0", "arg_1.notjson"],
		"no_txt": ["arg_0", "arg_1.json", "arg_2.nottxt"],
		"not_correct": ["arg_0", "arg_1.json", "arg_2.nottxt", "arg_3", "arg_4", "arg_5"],
		"correct": ["arg_0", "arg_1.json", "arg_2.txt", "arg_3", "arg_4", "arg_5"]
	}

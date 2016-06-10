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

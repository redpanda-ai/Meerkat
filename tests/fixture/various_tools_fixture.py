"""Fixtures for test_various_tools"""

from queue import Queue
from elasticsearch import Elasticsearch

def get_params_dict():
	"""Return a params dictionary"""
	return {
		"correct_format": {
			"input": {
				"hyperparameters": "tests/fixture/correct_format.json"
			}
		},
		"not_found": {
			"input": {
				"hyperparameters": "tests/missing.json"
			}
		}
	}

def get_es_connection(cluster):
	"""Return an es connection"""
	es_connection = Elasticsearch([cluster])
	return es_connection

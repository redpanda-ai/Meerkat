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

def get_dict_list():
	dict_list = [{'first_name,last_name,address,city,state,zip_code':\
				  'Tyrese,Hirthe,1404 Turner Ville,Strackeport,NY,19106-8813'},\
				 {'first_name,last_name,address,city,state,zip_code':\
				  'Jules,Dicki,2410 Estella Cape Suite 061,Lake Nickolasville,ME,00621-7435'},\
				 {'first_name,last_name,address,city,state,zip_code':\
				  'Dedric,Medhurst,6912 Dayna Shoal,Stiedemannberg,SC,43259-2273'}]
	return dict_list

def get_hyperparameters():
	return {
		"es_result_size": "45",
		"z_score_threshold": "2.857",
		"raw_score_threshold": "1.000",
		"good_description" : "2",
 		"boost_labels" : ["standard_fields"],
		"boost_vectors" :  {
			"address": [0.541],
			"address_extended": [1.282],
			"admin_region": [0.69],
			"category_labels": [1.319],
			"chain_name": [0.999],
			"email": [0.516],
			"internal_store_number": [1.9],
			"locality": [1.367],
			"name": [2.781],
			"neighborhood": [0.801],
			"po_box": [1.292],
			"post_town": [0.577],
			"postcode": [0.914],
			"region": [1.685],
			"tel": [0.597]
		}
	}

def get_boost():
	return {
		"raw_score_threshold": ["1.000"],
 		"boost_labels" : [["standard_fields"]],
		"boost_vectors" :  [{
			"address": [0.541],
			"address_extended": [1.282],
			"admin_region": [0.69],
			"category_labels": [1.319],
			"chain_name": [0.999],
			"email": [0.516],
			"internal_store_number": [1.9],
			"locality": [1.367],
			"name": [2.781],
			"neighborhood": [0.801],
			"po_box": [1.292],
			"post_town": [0.577],
			"postcode": [0.914],
			"region": [1.685],
			"tel": [0.597]
		}]
	}

def get_non_boost():
	return {
		"es_result_size": "45",
		"z_score_threshold": "2.857",
		"good_description" : "2",
	}

def get_boost_labels():
	return ["standard_fields"]

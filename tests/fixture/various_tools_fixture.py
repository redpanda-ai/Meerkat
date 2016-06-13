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
	dict_list = [{'first_name,last_name,address,city,state,zip_code': 'Tyrese,Hirthe,1404 Turner Ville,Strackeport,NY,19106-8813'},\
				 {'first_name,last_name,address,city,state,zip_code': 'Jules,Dicki,2410 Estella Cape Suite 061,Lake Nickolasville,ME,00621-7435'},\
				 {'first_name,last_name,address,city,state,zip_code': 'Dedric,Medhurst,6912 Dayna Shoal,Stiedemannberg,SC,43259-2273'}]
	return dict_list

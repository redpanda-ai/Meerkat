import csv
import sys
import json
import logging
import yaml
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers

logging.config.dictConfig(yaml.load(open('meerkat/logging.yaml', 'r')))
logger = logging.getLogger('tools')

endpoint = 'search-agg-and-factual-aq75mydw7arj5htk3dvasbgx4e.us-west-2.es.amazonaws.com'
host = [{'host': endpoint, 'port': 80}]
index_name, index_type = 'agg_index', 'agg_type'
es = Elasticsearch(host)

def pprint(dictionary, header=""):
	"""Simple function to pretty print a dictionary"""
	logger.info("{0}\n{1}".format(header, json.dumps(dictionary, sort_keys=True, indent=4)))

def search(query, **kwargs):
	"""Performs an Elasticsearch 'search'"""
	pprint(query, header="Query:")
	kwargs["index"] = index_name
	kwargs["doc_type"] = index_type
	kwargs["body"] = query

	results = es.search(**kwargs)
	hit_total = results['hits']['total']
	pprint(results, header="Total Hits: {0}".format(hit_total))

	return hit_total, results

def match_all():
	"""Query used to ensure that there are documents in the index."""
	query = {"query": {"match_all": {}}}
	return search(query)

def search_index(name):
	"""Basic query with no rules."""
	query = {
		'query': {
			'query_string': {
				'query': name,
				'fields': ['list_name']
			},
		},
		'_source': ['list_name', 'address', 'city', 'state', 'zip_code',
			'phone_number', 'store_number']
	}
	return search(query)

def search_with_name_and_store(list_name, store_number):
	"""Finds the exact store or no store for a particular merchant and store number."""
	query = {
		'query': {
			'bool': {
				'must': [
					{'term': {'list_name': list_name }},
					{'match': {'store_number': store_number}}
				]
			}
		},
		'_source': ['list_name', 'address', 'city', 'state', 'zip_code',
			'phone_number', 'store_number']
	}
	return search(query)

def search_with_name_city_and_state(list_name, city, state):
	"""Finds multiple stores when provided a name, city and state.."""
	query = {
		'query': {
			'bool': {
				'must': [
					{'term': {'list_name': list_name}},
					{'term': {'city': city}},
					{'term': {'state': state}}
				]
			}
		},
		'_source': ['list_name', 'address', 'city', 'state', 'zip_code',
			'phone_number', 'store_number']
	}
	return search(query)

def clean_store_number(store_number):
	chars = list(store_number)
	number = ""
	for char in chars:
		if char.isdigit():
			number += char
	return number

def search_with_name_and_zip(list_name, zip_code, description):
	"""Find multiple stores with name and zip code, and check if store number in description"""
	query = {
		'query': {
			'bool': {
				'must': [
					{'term': {'list_name': list_name }},
					{'match': {'zip_code': zip_code}}
				]
			}
		},
		'_source': ['list_name', 'address', 'city', 'state', 'zip_code',
			'phone_number', 'store_number']
	}

	hits, results = search(query)
	for result in results['hits']['hits']:
		store_number = clean_store_number(result['_source']['store_number'])
		if description.find(store_number) != -1:
			logger.info('Find the store')
			logger.info(result)

if __name__ == '__main__':
	# search_with_name_and_zip("The Home Depot", "78745", "THE HOME DEPOT #6570     SUNSET VALLEYTX")
	# search_with_name_and_zip("Walgreens", "33180", "WALGREENS #4955          AVENTURA     FL")
	search_with_name_and_zip("Target", "60707", "TARGET        00019240   CHICAGO      IL")

	# search_with_name_and_store("The Home Depot", "6372")
	# search_with_name_and_store("Colonial Finance", "4672")
	# search_with_name_and_store("Advance America", "1623")

	# search_with_name_city_and_state("The Home Depot", "Fort Lauderdale", "FL")
	# search_with_name_city_and_state("Walmart", "Chicago", "IL")
	# search_with_name_city_and_state("Redken", "Springfield", "IL")

	# search_index("Target")

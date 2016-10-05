import csv
import sys
import pandas as pd
import json
import logging
import yaml
from elasticsearch import Elasticsearch
from elasticsearch import helpers

logging.config.dictConfig(yaml.load(open('meerkat/logging.yaml', 'r')))
logger = logging.getLogger('tools')

endpoint = 'search-agg-index-drnxobzbjwkomgpm5dnfqccypa.us-west-2.es.amazonaws.com'
host = [{'host': endpoint, 'port': 80}]
index_name, index_type = 'agg_index_20161003', 'agg_type'
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

def search_index(string):
	"""Basic query with no rules."""
	query = {
		'query': {
			'query_string': {
				'query': string,
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
					# The term query requires a precise match
					{'term': {'list_name': list_name }},
					# A match query uses an analyzer to match tokens
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

if __name__ == '__main__':
	search_with_name_and_store("Title Credit Finance", "GA1287")
	search_with_name_and_store("Colonial Finance", "4672")
	search_with_name_and_store("Advance America", "1623")

	search_with_name_city_and_state("Walmart", "Chicago", "IL")
	search_with_name_city_and_state("Redken", "Springfield", "IL")

	search_index("Target")

	match_all()

import csv
import sys
import pandas as pd
import json
import logging
from elasticsearch import Elasticsearch
from elasticsearch import helpers

endpoint = 'search-agg-index-drnxobzbjwkomgpm5dnfqccypa.us-west-2.es.amazonaws.com'
host = [{'host': endpoint, 'port': 80}]
index_name = 'agg_index_20161003_test_4'
index_type = 'agg_type_20161003_test_4'

def match_all():
	es = Elasticsearch(host)
	res = es.search(index=index_name, size=1000, body={"query": {"match_all": {}}})
	ans = res['hits']['hits']
	print(len(ans))
	print(ans[0])

def search_index(string):
	"""Basic query with no rules."""
	es = Elasticsearch(host)
	res = es.search(index=index_name, doc_type=index_type, body={
		'query': {
			'query_string': {
				'query': string,
				'fields': ['list_name']
			},
		},
		'_source': ['list_name', 'address', 'city', 'state', 'zip_code', 'phone_number', 'store_number']
	})

	ans = res['hits']['hits']
	print(len(ans))
	print(ans[0])
	return ans

def search_with_name_and_store(list_name, store_number):
	"""Finds the exact store or no store for a particular merchant and store number."""
	es = Elasticsearch(host)
	body = {
		'query': {
			'bool': {
				'must': [
					{'term': {'list_name': list_name}},
					{'match': {'list_name': list_name}},
					{'match': {'store_number': store_number}}
				]
			}
		},
		'_source': ['list_name', 'address', 'city', 'state', 'zip_code', 'phone_number', 'store_number']
	}
	print(body)
	res = es.search(index=index_name, doc_type=index_type, body=body)

	ans = res['hits']['hits']
	if len(ans) > 0:
		print(ans[0])
	else:
		print("No match")
	return ans

def search_with_name_city_and_state(list_name, city, state):
	"""Finds multiple stores when provided a name, city and state.."""
	es = Elasticsearch(host)
	body = {
		'query': {
			'bool': {
				'must': [
					{'match': {'list_name': list_name}},
					{'match': {'city': city}},
					{'match': {'state': state}}
				]
			}
		},
		'_source': ['list_name', 'address', 'city', 'state', 'zip_code', 'phone_number', 'store_number']
	}
	print("Body:\n {0}".format(body))
	res = es.search(index=index_name, doc_type=index_type, body=body)

	ans = res['hits']['hits']
	if len(ans) > 0:
		print(json.dumps(ans, sort_keys=True, indent=4, separators=(',', ': ')))
	else:
		print("No match")
	return ans




if __name__ == '__main__':
	list_name, store_number = 'Title Credit Finance ', 'GA1287'
	search_with_name_and_store(list_name, store_number)
	#FIXME: Exact matches are probably a good idea, since 
	list_name, store_number = 'Colonial Finance', '4672'
	search_with_name_and_store(list_name, store_number)
	#list_name, store_number = 'Finance', '4672'
	#search_with_name_and_store(list_name, store_number)
	#list_name, city, state = "Walmart", "Chicago", "IL"
	#search_with_name_city_and_state(list_name, city, state)
	#match_all()

import csv
import sys
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers

endpoint = 'search-agg-index-drnxobzbjwkomgpm5dnfqccypa.us-west-2.es.amazonaws.com'
host = [{'host': endpoint, 'port': 80}]
index_name = 'agg_index'
index_type = 'agg_type'

def match_all():
	es = Elasticsearch(host)
	res = es.search(index=index_name, size=1000, body={"query": {"match_all": {}}})
	ans = res['hits']['hits']
	print(len(ans))
	print(ans[0])

def search_index(string):
	es = Elasticsearch(host)
	res = es.search(index=index_name, doc_type=index_type, body={
		'query': {
			'query_string': {
				'query': string,
				'fields': ['\ufefflist_name']
			},
		},
		'_source': ['\ufefflist_name', 'address', 'city', 'state', 'zip_code', 'phone_number', 'store_number']
	})

	ans = res['hits']['hits']
	print(len(ans))
	print(ans[0])
	return ans

def search_with_name_and_store(list_name, store_number)
	list_name, store_number = 'Title Credit Finance', 'GA1287'
	res = es.search(index=index_name, doc_type=index_type, body={
		'query': {
			'bool': {
				'must': [
					{'match': {'\ufefflist_name': list_name}},
					{'match': {'store_number': store_number}}
				]
			}
		},
		'_source': ['\ufefflist_name', 'address', 'city', 'state', 'zip_code', 'phone_number', 'store_number']
	})

	ans = res['hits']['hits']
	print(len(ans))
	print(ans[0])
	return ans

if __name__ == '__main__':
	list_name, store_number = 'Title Credit Finance', 'GA1287'
	search_with_name_and_store(list_name, store_number)

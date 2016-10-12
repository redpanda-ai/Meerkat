import sys
import json
import numpy as np
import pandas as pd
from pprint import pprint
from elasticsearch import Elasticsearch
from meerkat.various_tools import z_score_delta

endpoint = 'search-agg-and-factual-aq75mydw7arj5htk3dvasbgx4e.us-west-2.es.amazonaws.com'
host = [{'host': endpoint, 'port': 80}]
index_name, index_type = 'agg_index', 'agg_type'
es = Elasticsearch(host)

def basic_search(list_name):
	"""Performs an Elasticsearch search"""
	query = {
		'query': {
			'bool': {
				'must': {'term': {'list_name': list_name}}
			}
		}
	}
	result = es.search(index=index_name, doc_type=index_type, body=query)
	if result['hits']['total'] > 0:
		print('The number of hits is {}'.format(result['hits']['total']))
		pprint(result['hits']['hits'][0])
	else:
		print('The number of hits is zero')

def create_must_query(list_name, state, zip_code):
	"""Create a must query"""
	must_query = []
	if list_name != '':
		must_query.append({'term': {'list_name': list_name}})
	if state != '':
		must_query.append({'match': {'state': state}})
	if zip_code != '':
		must_query.append({'match': {'zip_code': zip_code}})
	return must_query

def create_should_query(city, store_number, phone_number):
	"""Create a should query"""
	should_query = []
	if city != '':
		should_query.append({'match': {'city': city}})
	if store_number != '':
		should_query.append({'match': {'store_number': store_number}})
	if phone_number != '':
		should_query.append({'match': {'phone_number': phone_number}})
	return should_query

def create_bool_query(must, should):
	"""Create a bool query with must and should query"""
	attributes = ['list_name', 'address', 'city', 'state', 'zip_code', 'phone_number', 'store_number']
	bool_query = {
		'query': {
			'bool': {
				'must': must,
				'should': should
			}
		},
		'_source': attributes
	}
	return bool_query

def search_index(filename):
	"""Search agg index for the input file"""
	df = pd.read_csv(filename, sep='|')
	df = df.fillna('')
	data = df.to_dict(orient='records')

	data = json.loads(open('./CNN_Agg.txt').read())
	requests = []
	header = {'index': index_name, 'doc_type': index_type}

	for i in range(len(data)):
		tran = data[i]
		description, list_name = tran.get('DESCRIPTION', ''), tran.get('MERCHANT_NAME', '')
		city, state, zip_code = tran.get('LOCALITY', ''), tran.get('STATE', ''), tran.get('ZIP_CODE', '')
		store_number, phone_number = tran.get('STORE_NUMBER', ''), tran.get('PHONE_NUMBER', '')

		list_name = data[i].get('Agg_Name', '')
		must = create_must_query(list_name, state, zip_code)
		should = create_should_query(city, store_number, phone_number)
		query = create_bool_query(must, should)

		requests.extend([header, query])

		results = es.msearch(body=requests)['responses']
		hits = results[0]['hits']['total']
		print('hits: {}    agg: {}'.format(hits, data[i]['Agg_Name']))
		requests = list()

if __name__ == '__main__':
	search_index('./processed_credit.csv')
	# basic_search('Armani Exchange (A/X)')

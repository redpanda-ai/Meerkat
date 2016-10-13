import sys
import json
import logging
import numpy as np
import pandas as pd
from pprint import pprint
from elasticsearch import Elasticsearch
from scipy.stats.mstats import zscore

logging.getLogger().setLevel(logging.INFO)

endpoint = 'search-agg-and-factual-aq75mydw7arj5htk3dvasbgx4e.us-west-2.es.amazonaws.com'
host = [{'host': endpoint, 'port': 80}]
index_name, index_type = 'agg_index', 'agg_type'
es = Elasticsearch(host)

def basic_search(list_name):
	"""Performs an Elasticsearch search"""
	query = {
		'query': {
			'bool': {'must': {'term': {'list_name': list_name}}}
		}
	}
	result = es.search(index=index_name, doc_type=index_type, body=query)
	if result['hits']['total'] > 0:
		logging.info('The number of hits is {}'.format(result['hits']['total']))
		pprint(result['hits']['hits'][0])
	else:
		logging.warning('The number of hits is zero')

def create_must_query(list_name, state, zip_code):
	"""Create a must query"""
	must_query = []
	if len(list_name) == 1 and list_name[0] != '':
		must_query.append({'term': {'list_name': list_name[0]}})
	else:
		bool_query = {'bool': {
						'should': [],
						'minimum_should_match': 1}}
		for name in list_name:
			bool_query['bool']['should'].append({'term': {'list_name': name}})
		must_query.append(bool_query)

	if state != '':
		must_query.append({'match': {'state': state}})
	if zip_code != '':
		zip_query = {'zip_code': {'query': zip_code, 'fuzziness': 'AUTO', 'boost': 3}}
		must_query.append({'match': zip_query})
	return must_query

def create_should_query(city, store_number, phone_number):
	"""Create a should query"""
	should_query = []
	if city != '':
		should_query.append({'match': {'city': city}})
	if store_number != '':
		store = {'store_number': {'query': store_number, 'fuzziness': 'AUTO', 'boost': 3}}
		should_query.append({'match': store})
	if phone_number != '':
		phone = {'phone_number': {'query': phone_number, 'fuzziness': 'AUTO', 'boost': 2}}
		should_query.append({'match': phone})
	return should_query

def create_bool_query(must_query, should_query):
	"""Create a bool query with must and should query"""
	attributes = ['list_name', 'address', 'city', 'state', 'zip_code', 'phone_number', 'store_number']
	bool_query = {
		'query': {
			'bool': {
				'must': must_query,
				'should': should_query
			}
		},
		'size': 1000,
		'_source': attributes
	}
	return bool_query

def process_query_result(query_result):
	"""Process the query hits based on z scores"""
	hits_total = query_result['hits']['total']
	logging.info('The number of hits is: {}'.format(hits_total))

	if hits_total < 3:
		logging.warning('The number of hits is less than 3')
		return None

	hits_list = query_result['hits']['hits']
	scores = []
	for hit in hits_list:
		scores.append(hit['_score'])

	z_scores = zscore(scores)
	if z_scores[0] - z_scores[1] >= 0:
		logging.info('This query has a top hit based on z scores')
		return hits_list[0]['_source']

	logging.warning('No top hit based on z scores')
	return None

def enrich_transaction(trans, hit):
	"""Enrich the input transaction with agg index hit"""
	attributes = ['list_name', 'address', 'city', 'state', 'zip_code', 'phone_number', 'store_number']
	if hit is not None:
		trans['agg_search'] = {}
		for key in attributes:
			if hit.get(key, '') != '':
				trans['agg_search'][key] = hit.get(key, '')

		logging.info('This transaction has been enriched with agg index')
		pprint(trans)
	return trans

def search_index(filename):
	"""Enrich transactions with agg index"""
	data = json.loads(open(filename).read())

	requests = []
	header = {'index': index_name, 'doc_type': index_type}

	for i in range(len(data)):
		trans = data[i]
		list_name = [trans.get('Agg_Name', '')]
		if len(list_name) == 0 or list_name[0] == '':
			logging.critical('A transaction with valid agg name is required, skip this one')
			continue

		city, state, zip_code = trans.get('city', ''), trans.get('state', ''), trans.get('zip_code', '')
		store_number, phone_number = trans.get('store_number', ''), trans.get('phone_number', '')

		must_query = create_must_query(list_name, state, zip_code)
		should_query = create_should_query(city, store_number, phone_number)
		bool_query = create_bool_query(must_query, should_query)

		logging.info('The query for this transaction is:')
		pprint(bool_query)

		requests.extend([header, bool_query])

		result = es.msearch(body=requests)['responses'][0]
		hit = process_query_result(result)
		enriched_trans = enrich_transaction(trans, hit)
		requests = []

if __name__ == '__main__':
	search_index('./CNN_Agg.txt')
	# basic_search('Starbucks US')

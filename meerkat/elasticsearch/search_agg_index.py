import sys
import json
import logging
import numpy as np
import pandas as pd
#from pprint import pprint
from elasticsearch import Elasticsearch
from scipy.stats.mstats import zscore

logging.getLogger().setLevel(logging.CRITICAL)

endpoint = 'search-agg-factual-nuz5jggrftlzjd5f7c2ehkmhlu.us-west-2.es.amazonaws.com'
host = [{'host': endpoint, 'port': 80}]
index_name, index_type = 'agg_index_10272016', 'agg_type_10272016'
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
		#pprint(result['hits']['hits'][0])
	else:
		logging.warning('The number of hits is zero')

def create_must_and_should(list_name, city, state, zip_code, store_number, phone_number, params):
	"""Create must and should query"""
	must_query, should_query = [], []

	# list_name always stays in must query
	if len(list_name) == 1 and list_name[0] != '':
		must_query.append({'term': {'list_name': list_name[0]}})
	else:
		bool_query = {'bool': {
						'should': [],
						'minimum_should_match': 1}}
		for name in list_name:
			bool_query['bool']['should'].append({'term': {'list_name': name}})
		must_query.append(bool_query)

	# city and state always stay in must query
	if city != '':
		must_query.append({'match': {'city': city}})
	if state != '':
		must_query.append({'match': {'state': state}})

	has_valid_store = False

	# store number always stays in should query
	if store_number != '':
		store = {'store_number': {'query': store_number, 'fuzziness': 'AUTO', 'boost': params['boost']['store_number']}}
		should_query.append({'match': store})
		has_valid_store = True

	# zip_code stays in should query if store number exists, otherwise stays in must query with ZERO fuzziness
	if zip_code != '':
		if has_valid_store == True:
			zip_query = {'zip_code': {'query': zip_code, 'fuzziness': 'AUTO', 'boost': params['boost']['zip_code']}}
			should_query.append({'match': zip_query})
		else:
			zip_query = {'zip_code': {'query': zip_code, 'fuzziness': 0, 'boost': params['boost']['zip_code']}}
			must_query.append({'match': zip_query})

	# phone number always stays in should query
	if phone_number != '':
		phone = {'phone_number': {'query': phone_number, 'fuzziness': 'AUTO', 'boost': params['boost']['phone_number']}}
		should_query.append({'match': phone})

	return must_query, should_query

def create_bool_query(must_query, should_query):
	"""Create a bool query with must and should query"""
	attributes = ['list_name', 'address', 'city', 'state', 'zip_code', 'phone_number',\
					'store_number', 'latitude', 'longitude', 'source_url']

	bool_query = {
		'query': {
			'bool': {
				'must': must_query,
				'should': should_query
			}
		},
		'size': 10000,
		'_source': attributes
	}
	return bool_query

def process_query_result(trans, query_result, params):
	"""Process the query hits based on z scores"""
	hits_total = query_result['hits']['total']

	# 0 hit
	if hits_total == 0:
		logging.critical('The number of hits is 0')
		return None

	# 1 hit
	if hits_total == 1:
		logging.critical('The number of hits is 1')
		if query_result['hits']['hits'][0]['_score'] >= params['threshold']['raw_score']:
			return query_result['hits']['hits'][0]
		else:
			return None

	des = trans.get('description', '')

	# 2 hits
	if hits_total == 2:
		logging.critical('The number of hits is 2')
		first, second = query_result['hits']['hits'][0], query_result['hits']['hits'][1]
		first_store = first['_source'].get('store_number', '')
		if first_store.startswith('T'):
			first_store = first_store[1:]

		second_store = second['_source'].get('store_number', '')
		if second_store.startswith('T'):
			second_store = second_store[1:]

		if first_store != '' and des.find(first_store) != -1:
			logging.critical('Find a store number in description')
			return first
		if second_store != '' and des.find(second_store) != -1:
			logging.critical('Find a store number in description')
			return second

		if first['_score'] - second['_score'] >= params['threshold']['z_score']:
			return first
		else:
			return None

	# 3 or more hits
	hits_list = query_result['hits']['hits']
	scores = []

	for hit in hits_list:
		scores.append(hit['_score'])

		store_number = hit['_source'].get('store_number', '')
		if store_number.startswith('T'):
			store_number = store_number[1:]
		if store_number.find('-') != -1:
			store_number = store_number.split('-')[0]
		if store_number != '' and des.find(store_number) != -1:
			logging.critical('Found a store number in description')
			return hit

	z_scores = zscore(scores)
	if z_scores[0] - z_scores[1] >= params['threshold']['z_score']:
		logging.info('This query has a top hit based on z scores')
		return hits_list[0]

	logging.critical('No top hit based on z scores')
	return None

def enrich_transaction(trans, hit):
	"""Enrich the input transaction with agg index hit"""
	attributes = ['list_name', 'address', 'city', 'state', 'zip_code', 'phone_number',\
					'store_number', 'latitude', 'longitude', 'source_url']

	if hit is not None:
		trans['agg_search'] = {}
		for key in attributes:
			if hit['_source'].get(key, '') != '':
				trans['agg_search'][key] = hit['_source'].get(key, '')

		logging.critical('This transaction has been enriched with agg index')
		if hit['_score'] < 2.0:
			logging.critical('The score for this transaction is less than 2.0')
			logging.critical(trans)
	return trans

def search_agg_index(data, params=None):
	"""Enrich transactions with agg index"""
	if params is None:
		params = json.loads(open('./meerkat/web_service/config/hyperparameters/search_agg_index_config.json').read())
	#pprint(params)

	requests = []
	header = {'index': index_name, 'doc_type': index_type}

	for i in range(len(data)):
		trans = data[i]
		logging.info('The transaction is:')
		#pprint(trans)

		list_name = trans.get('Agg_Name', [])
		if len(list_name) == 0 or list_name[0] == '':
			logging.critical('A transaction with valid agg name is required, skip this one')
			continue

		city, state, zip_code = trans.get('city', ''), trans.get('state', ''), trans.get('postal_code', '')
		phone_number, store_number = trans.get('phone_number', ''), trans.get('store_number', '')

		must_query, should_query = create_must_and_should(list_name, city, state, zip_code, store_number, phone_number, params)
		bool_query = create_bool_query(must_query, should_query)

		logging.info('The query for this transaction is:')
		#pprint(bool_query)

		requests.extend([header, bool_query])

		result = es.msearch(body=requests)['responses'][0]
		#pprint(result)
		hit = process_query_result(trans, result, params)
		enriched_trans = enrich_transaction(trans, hit)
		#pprint(enriched_trans)
		requests = []

if __name__ == '__main__':
	data = json.loads(open('./agg_input.json').read())
	data = list(np.random.permutation(data))
	search_agg_index(data)
	sys.exit()
	with open('./agg_output.json', 'w') as outfile:
		json.dump(data, outfile, indent=4, sort_keys=True)

import csv
import sys
import json
import logging
import yaml
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from meerkat.various_tools import z_score_delta

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

def search_with_name(name):
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

	hits, results = search(query)
	scores = []
	for result in results['hits']['hits']:
		scores.append(result['_score'])
	z_score = z_score_delta(scores)
	if z_score and z_score > 0.5:
		return True, results[0]

	return False, None

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
	hits, results = search(query)
	if hits > 0:
		return True, results['hits']['hits'][0]
	return False, None

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
			return True, result

	scores = []
	for result in results['hits']['hits']:
		scores.append(result['_score'])
	z_score = z_score_delta(scores)
	if z_score and z_score > 0.5:
		return True, results[0]

	return False, None

def search_with_name_city_and_state(list_name, city, state, description):
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

	hits, results = search(query)
	for result in results['hits']['hits']:
		store_number = clean_store_number(result['_source']['store_number'])
		if description.find(store_number) != -1:
			return True, result

	scores = []
	for result in results['hits']['hits']:
		scores.append(result['_score'])
	z_score = z_score_delta(scores)
	if z_score and z_score > 0.5:
		return True, results[0]

	return False, None

def process_trans(filename):
	df = pd.read_csv(filename, sep='|')
	df['ADDRESS'] = ''

	count = 0
	for index, row in df.iterrows():
		count += 1
		if count == 200: break

		print(row)
		des, name = row['DESCRIPTION'], row['MERCHANT_NAME']
		city, state = row['LOCALITY'], row['STATE']
		store, phone = row['STORE_NUMBER'], row['PHONE_NUMBER']

		tran = row.notnull()
		flag_des, flag_name = tran['DESCRIPTION'], tran['MERCHANT_NAME']
		flag_city, flag_state = tran['LOCALITY'], tran['STATE']
		flag_store, flag_phone = tran['STORE_NUMBER'], tran['PHONE_NUMBER']

		print(tran)
		if flag_name and flag_city and flag_state:
			flag, result = search_with_name_city_and_state(name, city, state, des)
			if flag:
				df.set_value(index, 'ADDRESS', result['_source']['address'])
				continue
		if flag_name and flag_store:
			flag, result = search_with_name_and_store(name, store)
			if flag:
				df.set_value(index, 'ADDRESS', result['_source']['address'])
				continue

		df.set_value(index, 'ADDRESS', "No Address Found")

	df.to_csv('./test.csv', sep='|', index=False, header=True)
	print('Done')

if __name__ == '__main__':
	process_trans('./processed_credit.csv')

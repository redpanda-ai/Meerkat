import csv
import sys
import logging
import yaml
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers


logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('tools')

endpoint = 'search-agg-index-drnxobzbjwkomgpm5dnfqccypa.us-west-2.es.amazonaws.com'
host = [{'host': endpoint, 'port': 80}]
index_name = 'agg_index_20161003'
index_type = 'agg_type_20161003'

def match_all():
	es = Elasticsearch(host)
	res = es.search(index=index_name, size=1000, body={"query": {"match_all": {}}})
	ans = res['hits']['hits']
	logger.info(len(ans))
	logger.info(ans[0])

def outer_func(filename):
	es = Elasticsearch(host)
	if es.indices.exists(index_name):
		res = es.indices.delete(index=index_name)

	chunk_count = 0
	chunksize = 10000
	reader = pd.read_csv(filename, chunksize=chunksize)
	for chunk in reader:
		#logger.info(chunk)
		logger.info("Chunk {0}".format(chunk_count))
		build_index(chunk, es, chunk_count, chunksize)
		chunk_count += 1

def build_index(df, es, chunk_count, chunksize):
	for column in df.columns:
		df[column] = df[column].astype('str')

	data = df.to_dict(orient='records')
	logger.info(len(data))


	actions = []
	offset = chunk_count * chunksize
	for i in range(len(data)):
		action = {
			'_index': index_name,
			'_type': index_type,
			'_id': offset + i,
			'_source': data[i]
		}
		actions.append(action)

	if len(actions) > 0:
		helpers.bulk(es, actions)

if __name__ == '__main__':
	outer_func('./selected-lists-5224.csv')
	#match_all()

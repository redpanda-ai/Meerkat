import csv
import sys
import logging
import yaml
import json
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers


logging.config.dictConfig(yaml.load(open('meerkat/logging.yaml', 'r')))
logger = logging.getLogger('tools')

endpoint = 'search-agg-index-drnxobzbjwkomgpm5dnfqccypa.us-west-2.es.amazonaws.com'
host = [{'host': endpoint, 'port': 80}]
index_name = 'agg_index_20161003_test_4'
index_type = 'agg_type_20161003_test_4'

def match_all():
	es = Elasticsearch(host)
	res = es.search(index=index_name, size=1000, body={"query": {"match_all": {}}})
	ans = res['hits']['hits']
	logger.info(len(ans))
	logger.info(json.dumps(ans[0], indent=4, sort_keys=True, separators=(",",":")))

def outer_func(filename):
	es = Elasticsearch(host)
	if es.indices.exists(index_name):
		res = es.indices.delete(index=index_name)

	#set all data types to "str" in the reader, so that nothing is mangled
	reader = pd.read_csv(filename, chunksize=10)
	test_chunk = reader.get_chunk(0)
	dtype = {}
	for column in test_chunk.columns:
		dtype[column] = "str"

	chunk_count = 0
	chunksize = 10000
	reader = pd.read_csv(filename, chunksize=chunksize, dtype=dtype)
	for chunk in reader:
		logger.info("Chunk {0}".format(chunk_count))
		build_index(chunk, es, chunk_count, chunksize)
		chunk_count += 1

def build_index(df, es, chunk_count, chunksize):
	#This ensures that all columns are cast as strings
	for column in df.columns:
		df[column] = df[column].astype('str')

	data = df.to_dict(orient='records')

	actions = []
	offset = chunk_count * chunksize
	
	for i in range(len(data)):
		#It may be preferable to use pandas to alter the dataframe ahead of time.
		#For now, I'm just removing all NaN values
		for j in list(data[i]):
			if data[i][j] == "nan":
				del data[i][j]
		action = {
			'_index': index_name,
			'_type': index_type,
			'_id': offset + i,
			'_source': data[i]
		}
		actions.append(action)

	logging.info(actions)
	if len(actions) > 0:
		helpers.bulk(es, actions)

if __name__ == '__main__':
	outer_func('./selected-lists-5224.csv')
	#Uncomment the following to look at a sample result of a match_all query
	#match_all()

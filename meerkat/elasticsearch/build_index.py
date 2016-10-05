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
index_name, index_type = 'agg_index_20161003', 'agg_type'

def real_main(filename):
	"""This is the main function, which creates the index, type_mapping, and
	then populates the index."""
	es = Elasticsearch(host)
	if es.indices.exists(index_name):
		logger.warning("Deleting existing index.")
		res = es.indices.delete(index=index_name)

	#create an index with a few hints for the type_mapping schema
	not_analyzed_string = { "index": "not_analyzed", "type" : "string"}
	type_mapping = {
		"mappings": {
			index_type: {
				"_source": {
					"enabled" : True
				},
				"properties" : {
					"list_name": not_analyzed_string,
					"city": not_analyzed_string,
					"state": not_analyzed_string
				}
			}
		}
	}
	logger.info("Creating index with type mapping")
	es.indices.create(index=index_name, body=type_mapping)
	logger.info("Index created")

	#set all data types to "str" in the reader, so that nothing is mangled
	reader = pd.read_csv(filename, chunksize=10)
	test_chunk = reader.get_chunk(0)
	dtype = {}
	for column in test_chunk.columns:
		dtype[column] = "str"

	#build the index, one chunk at a time
	chunk_count, chunksize = 0, 10000
	reader = pd.read_csv(filename, chunksize=chunksize, dtype=dtype)
	for chunk in reader:
		logger.info("Chunk {0}".format(chunk_count))
		load_dataframe_into_index(chunk, es, chunk_count, chunksize)
		chunk_count += 1

def load_dataframe_into_index(df, es, chunk_count, chunksize):
	"""Bulk Load a dataframe into the index"""
	#This ensures that all columns are cast as strings
	for column in df.columns:
		df[column] = df[column].astype('str')

	data = df.to_dict(orient='records')
	#Strip whitespace, which may interfere with a term query
	df.apply(lambda x: x.str.strip(), axis=1)

	actions = []
	offset = chunk_count * chunksize
	
	for i in range(len(data)):
		#Removing all NaN values
		for j in list(data[i]):
			if data[i][j] == "nan":
				del data[i][j]
		#creating actions for a bulk load operation
		action = {
			'_index': index_name,
			'_type': index_type,
			'_id': offset + i,
			'_source': data[i]
		}
		actions.append(action)

	#Bulk load the index
	if len(actions) > 0:
		helpers.bulk(es, actions)

if __name__ == '__main__':
	real_main('./selected-lists-5224.csv')
	#Uncomment the following to look at a sample result of a match_all query
	#match_all()

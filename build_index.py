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

def build_index(filename):
	df = pd.read_csv(filename)
	for column in df.columns:
		df[column] = df[column].astype('str')

	data = df.to_dict(orient='records')
	print(len(data))

	es = Elasticsearch(host)

	if es.indices.exists(index_name):
		res = es.indices.delete(index=index_name)

	actions = []
	for i in range(len(data)):
		action = {
			'_index': index_name,
			'_type': index_type,
			'_id': i,
			'_source': data[i]
		}
		actions.append(action)

	if len(actions) > 0:
		helpers.bulk(es, actions)

if __name__ == '__main__':
	build_index('./agg_raw.csv')
	match_all()

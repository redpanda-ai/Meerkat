#!/bin/bash

curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.search.size" : 20, "threadpool.search.queue_size": 50 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.bulk.size" : 20, "threadpool.bulk.queue_size": 1000 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.index.size" : 30, "threadpool.index.queue_size": 1000 } }'


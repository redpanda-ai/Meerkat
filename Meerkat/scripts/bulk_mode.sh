#!/bin/bash

curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.search.size" : 4, "threadpool.search.queue_size": 500 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.bulk.size" : 8, "threadpool.bulk.queue_size": 4000 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.index.size" : 32, "threadpool.index.queue_size": 5000 } }'


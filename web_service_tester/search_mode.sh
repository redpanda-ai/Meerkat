#!/bin/bash

curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.search.size" : 32, "threadpool.search.queue_size": 4000 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.bulk.size" : 1, "threadpool.bulk.queue_size": 200 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.index.size" : 8, "threadpool.index.queue_size": 1000 } }'


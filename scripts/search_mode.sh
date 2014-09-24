#!/bin/bash

curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.search.size" : 600, "threadpool.search.queue_size": 3000 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.bulk.size" : 10, "threadpool.bulk.queue_size": 200 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "threadpool.index.size" : 50, "threadpool.index.queue_size": 500 } }'


#!/bin/bash

ssh -i ~/.ssh/meerkat.pem ${1} "sed -i 's/solo/${1}/' /etc/elasticsearch/elasticsearch.yml"

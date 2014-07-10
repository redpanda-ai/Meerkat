#!/bin/bash

ssh -i ~/.ssh/meerkat.pem ${1} "sed -i 's/__X__/${1}/' /etc/elasticsearch/elasticsearch.yml"

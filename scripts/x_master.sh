#!/bin/bash
ssh -i ~/.ssh/meerkat.pem ${1} "sed -i '64 s/#//' /etc/elasticsearch/elasticsearch.yml"
ssh -i ~/.ssh/meerkat.pem ${1} "sed -i '65 s/#//' /etc/elasticsearch/elasticsearch.yml"


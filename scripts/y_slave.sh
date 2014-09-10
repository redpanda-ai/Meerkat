#!/bin/bash
ssh -i ~/.ssh/meerkat.pem ${1} "sed -i '58 s/#//' /etc/elasticsearch/elasticsearch.yml"
ssh -i ~/.ssh/meerkat.pem ${1} "sed -i '59 s/#//' /etc/elasticsearch/elasticsearch.yml"


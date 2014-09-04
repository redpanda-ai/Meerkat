#!/bin/bash

ssh -i ~/.ssh/meerkat.pem ${1} "chown -R elasticsearch.elasticsearch /mnt/ephemeral/elasticsearch/"

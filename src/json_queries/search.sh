#!/bin/bash

curl -s -XPOST brainstorm8:9200/merchants/merchant/_search -d @${1}

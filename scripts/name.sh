#!/bin/bash

curl -s s16:9200/_nodes/stats?pretty=true | grep -E '\".{22}\" \: \{'

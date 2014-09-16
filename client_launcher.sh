#!/bin/bash

if [ "$#" -ne 3 ]; then
	echo "You must provide 3 arguments: container, filter, and config_name"
	exit 1
fi

cat config/template.json | sed -e "s/__CONTAINER/${1}/g" | sed -e "s/__FILTER/${2}/g" > config/${3}.json
time nohup python3.3 -u -m meerkat config/${3}.json > ${3}.out 2>&1&


#!/bin/bash

for i in `seq 1 1000000`;
do
	clear
	date
	ps -ef | grep python3.3 | sed -e "s/.*\(python.*\)/\1/g"
	wc -l /data/1/output/*.txt
	sleep 5
done

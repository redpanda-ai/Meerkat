#!/bin/bash

lines=""
for i in `seq 1 1000000`;
do
	#clear
	#date
	#ps -ef | grep python3.3 | sed -e "s/.*\(python.*\)/\1/g" | awk ' { print $5 } '
	#wc -l /data/0/input/*.txt
	lines=` wc -l /data/1/output/*.txt | grep total`
	echo -e "Lines are ${lines}"
	sleep 5
done

#!/bin/bash

lines=""
#total="174304688"
total=` wc -l /data/0/input/*.txt | grep total | awk ' { print $1 } '`

for i in `seq 1 1000000`;
do
	#clear
	#date
	#ps -ef | grep python3.3 | sed -e "s/.*\(python.*\)/\1/g" | awk ' { print $5 } '
	#wc -l /data/0/input/*.txt
	lines=` wc -l /data/1/output/*.txt | grep total | awk ' { print $1 } '`
	percent=` bc -l <<< "scale=4; $lines / $total"`
	echo -e "Lines are ${percent}: ${lines} / ${total}"
	sleep 5
done

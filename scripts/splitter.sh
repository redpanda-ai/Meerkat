#!/bin/bash

tail -n +2 ${1} | split -a 3 -d -l 1000000 - ${1}.
denom=` ls -Fal ${1}.* | awk ' { print $9 } ' | cut -d '.' -f 3 | tail -n 1 `
for file in ${1}.*
do
	head -n 1 ${1} > tmp_file
	cat $file >> tmp_file
	mv -f tmp_file $file
	mv -f $file $file.$denom
	gzip $file.$denom
	aws s3 cp --sse $file.$denom s3://yodleeprivate/ctprocessed/gpanel/bank/
	rm $file.$denom
done

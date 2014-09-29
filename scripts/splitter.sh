#!/bin/bash

#[root@ip-172-31-23-58 chunk]# ls -Fal 20140219_GPANEL_BANK.txt.* | awk ' { print $9 }'
#20140219_GPANEL_BANK.txt.034
#[root@ip-172-31-23-58 chunk]# ls -Fal 20140219_GPANEL_BANK.txt.* | awk ' { print $9 }' | cut -d '.' -f 3
#034
#[root@ip-172-31-23-58 chunk]# ls 20140219_GPANEL_BANK.txt.034 | cut -d '_' -f 1 
#20140219
#[root@ip-172-31-23-58 chunk]# ls 20140219_GPANEL_BANK.txt.034 | cut -d '_' -f 1 2
#cut: 2: No such file or directory
#[root@ip-172-31-23-58 chunk]# ls 20140219_GPANEL_BANK.txt.034 | cut -d '_' -f 2,3
#GPANEL_BANK.txt.034
#[root@ip-172-31-23-58 chunk]# ls 20140219_GPANEL_BANK.txt.034 | cut -d '_' -f 2,3 | cut -d '.' -f 3
#034



tail -n +2 ${1} | split -a 3 -d -l 1000000 - ${1}.
denom=` ls -Fal ${1}.* | awk ' { print $9 } ' | cut -d '.' -f 3 | tail -n 1 `
echo -e "Denom is $denom"
for file in ${1}.*
do
	prefix=` ls -Fal $file | awk ' { print $9 } ' | cut -d '_' -f 1 `
	numer=` ls -Fal $file | awk ' { print $9 } ' | cut -d '.' -f 3 `
	suffix=` ls -Fal $file | awk ' { print $9 } ' | cut -d '_' -f 2,3 | cut -d '.' -f 1,2`
	total_file=$prefix.$numer.$denom'_'$suffix

	head -n 1 ${1} > tmp_file
	cat $file >> tmp_file
	mv -f tmp_file $file
	mv -f $file $total_file
	echo -e "Gzipping $total_file"
	gzip $total_file
	echo -e "Shipping ${total_file}.gz"
	aws s3 cp --sse ${total_file}.gz s3://yodleeprivate/ctprocessed/gpanel/bank/
	echo -e "Removing ${total_file}.gz"
	rm ${total_file}.gz
done

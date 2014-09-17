ec2-describe-instances > file_1
cat file_1 | grep INSTANCE | grep sg-2db53f48 > file_2
cat file_2 | grep 2014-09-17 > file_3
cat file_3 | awk ' { print $15 } ' > file_4
sort -t . -k 1,1n -k 2,2n -k 3,3n -k 4,4n file_4 > file_5

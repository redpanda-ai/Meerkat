ec2-describe-instances > file_1
cat file_1 | grep INSTANCE | grep sg-2db53f48 > file_2
cat file_2 | grep 2014-09-10 > file_3
cat file_3 | awk ' { print $15 } ' > file_4

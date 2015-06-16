for i in `seq 1 500`;
do
	# piping standard error into standard output
	./web_service_correct.sh $1 >> 500.log 2>&1
done

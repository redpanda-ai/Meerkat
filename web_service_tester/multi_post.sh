for i in `seq 1 $3`;
do
	# piping standard error into standard output
	./single_post.sh $1 $2 >> multi_post.log 2>&1
done

prefix="x"
node_start=10001
node_end=10526



#echo -e "Nodes are:"
nodes=()
for ((a=${node_start}; a <= ${node_end} ; a++))
do
	nodes+=("${prefix}${a}")
	#echo -e "${prefix}${a}"
done

SSH_OPTIONS="-i /root/.ssh/meerkat.pem -o StrictHostKeyChecking=no"
echo -e "Inspecting"
for n in "${nodes[@]}"
do
	echo -e "${n}"
	ssh ${SSH_OPTIONS} ${n} "ls -FalR /data" | grep "factual_index\/[0-9]*\:" | sed -e 's/.*\/\(.*\):/\1/' | sort -n | uniq
done


prefix="n"
node_start=10001
node_end=10039

#echo -e "Nodes are:"
nodes=()
for ((a=${node_start}; a <= ${node_end} ; a++))
do
	nodes+=("${prefix}${a}")
	#echo -e "${prefix}${a}"
done

KEY="/root/.ssh/meerkat.pem"
echo -e "Inspecting"
for n in "${nodes[@]}"
do
	echo -e "${n}"
	ssh -i ${KEY} ${n} "ls -FalR /data" | grep "factual_index\/[0-9]*\:" | sed -e 's/.*\/\(.*\):/\1/' | sort -n | uniq
done

prefix="b"
node_start=10001
node_end=10068

#echo -e "Nodes are:"
nodes=()
for ((a=${node_start}; a <= ${node_end} ; a++))
do
	nodes+=("${prefix}${a}")
	#echo -e "${prefix}${a}"
done

KEY="/root/.ssh/meerkat.pem"
#echo -e "Inspecting"
for n in "${nodes[@]}"
do
	echo -e "${n}"
	ssh -i ${KEY} ${n} "ls -FalR /data" | grep "factual_index\/[0-9]*\:" | sed -e 's/.*\/\(.*\):/\1/' | sort -n | uniq
done

prefix="spot"
node_start=10001
node_end=10699

#echo -e "Nodes are:"
nodes=()
for ((a=${node_start}; a <= ${node_end} ; a++))
do
	nodes+=("${prefix}${a}")
	#echo -e "${prefix}${a}"
done

KEY="/root/.ssh/meerkat.pem"
#echo -e "Inspecting"
for n in "${nodes[@]}"
do
	echo -e "${n}"
	ssh -i ${KEY} ${n} "ls -FalR /data" | grep "factual_index\/[0-9]*\:" | sed -e 's/.*\/\(.*\):/\1/' | sort -n | uniq
done



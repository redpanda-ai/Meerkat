prefix="n"
node_start=21
node_end=100
slave_start=21
slave_end=100
master_start=1
master_end=0
echo -e "Nodes are:"
nodes=()
for ((a=${node_start}; a <= ${node_end} ; a++))
do
	nodes+=("${prefix}${a}")
	echo -e "${prefix}${a}"
done

echo -e "Slaves are:"
slaves=()
for ((b=${slave_start}; b <= ${slave_end} ; b++))
do
	slaves+=("${prefix}${b}")
	echo -e "${prefix}${b}"
done

echo -e "Masters are:"
masters=()
for ((c=${master_start}; c <= ${master_end} ; c++))
do
	masters+=("${prefix}${c}")
	echo -e "${prefix}${c}"
done

KEY="/root/.ssh/meerkat.pem"

echo -e "Blanking cluster"
for n in "${nodes[@]}"
do
	scp -i ${KEY} /etc/elasticsearch/elasticsearch.yml ${n}:/etc/elasticsearch/elasticsearch.yml
done
echo -e "Naming Nodes"
for m in "${nodes[@]}"
do
	./rename.sh ${m}
done
echo -e "Setting Masters"
for i in "${masters[@]}"
do
	./y_slave.sh ${i}
done
echo -e "Setting Slaves"
for j in "${slaves[@]}"
do
	./y_slave.sh ${j}
done
echo -e "Mounting EBS to /data"
for k in "${nodes[@]}"
do
	./prep_data.sh ${k}
done
echo -e "Mounting EBS to /data"
for p in "${nodes[@]}"
do
	ssh -i ${KEY} ${p} "service elasticsearch start"
done

echo -e "Complete."

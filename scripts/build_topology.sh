prefix="n"
master_start=10001
master_end=10002

slave_start=10003
slave_end=10020

node_start=10001
node_end=10020

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
	echo -e "Renaming ${prefix}${m}"
	ssh -i ${KEY} ${m} "sed -i 's/solo/${prefix}${m}/' /etc/elasticsearch/elasticsearch.yml"
done
echo -e "Setting Masters"
for i in "${masters[@]}"
do
	echo -e "master ${prefix}${i}"
	ssh -i ${KEY} ${i} "sed -i '64 s/#//' /etc/elasticsearch/elasticsearch.yml"
	ssh -i ${KEY} ${i} "sed -i '65 s/#//' /etc/elasticsearch/elasticsearch.yml"
done
echo -e "Setting Slaves"
for j in "${slaves[@]}"
do
	echo -e "slave ${prefix}${j}"
	ssh -i ${KEY} ${j} "sed -i '58 s/#//' /etc/elasticsearch/elasticsearch.yml"
	ssh -i ${KEY} ${j} "sed -i '59 s/#//' /etc/elasticsearch/elasticsearch.yml"
done
echo -e "Mounting EBS to /data"
for k in "${nodes[@]}"
do
	echo -e "Mounting EBS for ${prefix}${k}"
	ssh -i ${KEY} ${k} "mkfs -t ext4 /dev/xvdb"
	ssh -i ${KEY} ${k} "mount /dev/xvdb /data"
	ssh -i ${KEY} ${k} "df | grep 'data'"
	ssh -i ${KEY} ${k} "chown -R elasticsearch /data"
done
echo -e "Starting Elasticsearch"
for p in "${nodes[@]}"
do
	echo -e "Activating ${prefix}${p}"
	ssh -i ${KEY} ${p} "service elasticsearch start"
done

echo -e "Complete."

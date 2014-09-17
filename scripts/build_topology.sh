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
	echo -e "Renaming ${m}"
	ssh -i ${KEY} ${m} "sed -i 's/solo/${m}/' /etc/elasticsearch/elasticsearch.yml"
done
echo -e "Setting Masters"
for i in "${masters[@]}"
do
	echo -e "master ${i}"
	ssh -i ${KEY} ${i} "sed -i '64 s/#//' /etc/elasticsearch/elasticsearch.yml"
	ssh -i ${KEY} ${i} "sed -i '65 s/#//' /etc/elasticsearch/elasticsearch.yml"
done
echo -e "Setting Slaves"
for j in "${slaves[@]}"
do
	echo -e "slave ${j}"
	ssh -i ${KEY} ${j} "sed -i '58 s/#//' /etc/elasticsearch/elasticsearch.yml"
	ssh -i ${KEY} ${j} "sed -i '59 s/#//' /etc/elasticsearch/elasticsearch.yml"
done
echo -e "Mounting EBS to /data and starting Elasticsearch"
for k in "${nodes[@]}"
do
	echo -e "Activating ${k}"
	#ssh -i ${KEY} ${k} "mkfs -t ext4 /dev/xvdb"
	ssh -i ${KEY} ${k} "mkdir -p /data/0"
	ssh -i ${KEY} ${k} "mkdir -p /data/1"
	ssh -i ${KEY} ${k} "mount /dev/xvdb /data/0"
	ssh -i ${KEY} ${k} "mount /dev/xvdc /data/1"
	ssh -i ${KEY} ${k} "df | grep 'data'"
	ssh -i ${KEY} ${k} "mkdir -p /data/0/input"
	ssh -i ${KEY} ${k} "mkdir -p /data/0/output"
	ssh -i ${KEY} ${k} "mkdir -p /data/0/error"
	ssh -i ${KEY} ${k} "mkdir -p /data/1/input"
	ssh -i ${KEY} ${k} "mkdir -p /data/1/output"
	ssh -i ${KEY} ${k} "mkdir -p /data/1/error"
	ssh -i ${KEY} ${k} "mkdir -p /data/1/log/elasticsearch"

	ssh -i ${KEY} ${k} "chown -R elasticsearch /data"
	ssh -i ${KEY} ${k} "service elasticsearch start"
done

echo -e "Complete."

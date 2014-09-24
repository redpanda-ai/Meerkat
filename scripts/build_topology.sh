prefix=${1}
master_start=1
master_end=0

slave_start=10011
slave_end=10300

node_start=10011
node_end=10300

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
SSH_OPTIONS="-i /root/.ssh/meerkat.pem -o StrictHostKeyChecking=no"

echo -e "Blanking cluster"
for n in "${nodes[@]}"
do
	scp ${SSH_OPTIONS} /etc/elasticsearch/elasticsearch.yml ${n}:/etc/elasticsearch/elasticsearch.yml
done
echo -e "Naming Nodes"
for m in "${nodes[@]}"
do
	echo -e "Renaming ${m}"
	ssh ${SSH_OPTIONS} ${m} "sed -i 's/solo/${m}/' /etc/elasticsearch/elasticsearch.yml"
done
echo -e "Setting Masters"
for i in "${masters[@]}"
do
	echo -e "master ${i}"
	ssh ${SSH_OPTIONS} ${i} "sed -i '64 s/#//' /etc/elasticsearch/elasticsearch.yml"
	ssh ${SSH_OPTIONS} ${i} "sed -i '65 s/#//' /etc/elasticsearch/elasticsearch.yml"
done
echo -e "Setting Slaves"
for j in "${slaves[@]}"
do
	echo -e "slave ${j}"
	ssh ${SSH_OPTIONS} ${j} "sed -i '58 s/#//' /etc/elasticsearch/elasticsearch.yml"
	ssh ${SSH_OPTIONS} ${j} "sed -i '59 s/#//' /etc/elasticsearch/elasticsearch.yml"
done
echo -e "Mounting EBS to /data and starting Elasticsearch"
for k in "${nodes[@]}"
do
	echo -e "Activating ${k}"
	#ssh ${SSH_OPTIONS} ${k} "mkfs -t ext4 /dev/xvdb"
	ssh ${SSH_OPTIONS} ${k} "mkdir -p /data/0"
	ssh ${SSH_OPTIONS} ${k} "mkdir -p /data/1"
	ssh ${SSH_OPTIONS} ${k} "mount /dev/xvdb /data/0"
	ssh ${SSH_OPTIONS} ${k} "mount /dev/xvdc /data/1"
	ssh ${SSH_OPTIONS} ${k} "df | grep 'data'"
	ssh ${SSH_OPTIONS} ${k} "mkdir -p /data/0/input"
	ssh ${SSH_OPTIONS} ${k} "mkdir -p /data/0/output"
	ssh ${SSH_OPTIONS} ${k} "mkdir -p /data/0/error"
	ssh ${SSH_OPTIONS} ${k} "mkdir -p /data/1/input"
	ssh ${SSH_OPTIONS} ${k} "mkdir -p /data/1/output"
	ssh ${SSH_OPTIONS} ${k} "mkdir -p /data/1/error"
	ssh ${SSH_OPTIONS} ${k} "mkdir -p /data/1/log/elasticsearch"
	ssh ${SSH_OPTIONS} ${k} "chown -R elasticsearch /data"
	ssh ${SSH_OPTIONS} ${k} "service elasticsearch start"
done

echo -e "Complete."

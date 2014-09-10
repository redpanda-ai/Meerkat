echo -e "Blanking cluster"
./scp_all.sh /etc/elasticsearch/elasticsearch.yml 
echo -e "Naming Nodes"
./rename_all.sh
echo -e "Setting Masters"
./x_master.sh n01
./x_master.sh n02
echo -e "Setting Slaves"
./y_slave.sh n03
./y_slave.sh n04
./y_slave.sh n05
./y_slave.sh n06
./y_slave.sh n07
./y_slave.sh n08
./y_slave.sh n09
./y_slave.sh n10
./y_slave.sh n11
./y_slave.sh n12
./y_slave.sh n13
./y_slave.sh n14
./y_slave.sh n15
./y_slave.sh n16
./y_slave.sh n17
./y_slave.sh n18
./y_slave.sh n19
./y_slave.sh n20
echo -e "Mounting EBS to /data"
./prep_data.sh n01
./prep_data.sh n02
./prep_data.sh n03
./prep_data.sh n04
./prep_data.sh n05
./prep_data.sh n06
./prep_data.sh n07
./prep_data.sh n08
./prep_data.sh n09
./prep_data.sh n10
./prep_data.sh n11
./prep_data.sh n12
./prep_data.sh n13
./prep_data.sh n14
./prep_data.sh n15
./prep_data.sh n16
./prep_data.sh n17
./prep_data.sh n18
./prep_data.sh n19
./prep_data.sh n20
echo -e "Complete."

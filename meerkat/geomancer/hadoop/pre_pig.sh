export PIG_CLASSPATH=/opt/cloudera/parcels/CDH-4.4.0-1.cdh4.4.0.p0.39/lib/hcatalog/share/webhcat/svr/lib/hcatalog-core-0.5.0-cdh4.4.0.jar:\
/opt/cloudera/parcels/CDH-4.4.0-1.cdh4.4.0.p0.39/lib/hcatalog/share/hcatalog/hcatalog-pig-adapter-0.5.0-cdh4.4.0.jar:\
$HIVELIBS/hive-metastore-0.10.0-cdh4.4.0.jar:$HIVELIBS/libthrift-0.9.0-cdh4-1.jar:\
$HIVELIBS/hive-exec-0.10.0-cdh4.4.0.jar:$HIVELIBS/libfb303-0.9.0.jar:\
$HIVELIBS/jdo2-api-2.3-ec.jar:/etc/hive/conf:$HADOOP_CONF:\
$HIVELIBS/slf4j-api-1.6.4.jar

export PIG_OPTS=-Dhive.metastore.uris=thrift://hpdl-R306-16.yodlee.com:9083

export PIG_HEAPSIZE=4096

source /etc/hadoop/cdh5_hf_spark3.0
spark-submit \
--master yarn \
--deploy-mode cluster \
--num-executors 30 \
--executor-cores 4 \
--executor-memory 32G \
--driver-memory 12G \
--conf spark.driver.extraClassPath=hdfs://ns-hf/project/compass/compass/hive/udf/daas_tools-1.0-SNAPSHOT.jar \
--conf spark.driver.extraLibraryPath=/opt/hadoopsys/hadoop/lib/native \
--conf spark.driver.host=hfa-pro0207.hadoop.cpcc.iflyyun.cn \
source /etc/hadoop/cdh5_hf_spark3.0
spark-submit \
--master yarn \
--deploy-mode cluster \
--num-executors 30 \
--executor-cores 4 \
--executor-memory 32G \
--driver-memory 12G \
--conf spark.driver.extraClassPath=hdfs://ns-hf/project/compass/compass/hive/udf/daas_tools-1.0-SNAPSHOT.jar \
--conf spark.driver.extraLibraryPath=/opt/hadoopsys/hadoop/lib/native \
--conf spark.driver.host=hfa-pro0207.hadoop.cpcc.iflyyun.cn \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.eventLog.enabled=true \
--conf spark.executor.extraClassPath=hdfs://ns-hf/project/compass/compass/hive/udf/daas_tools-1.0-SNAPSHOT.jar \
--conf spark.executor.extraLibraryPath=/opt/hadoopsys/hadoop/lib/native \
--conf hive.metastore.uris=thrift://192.168.72.18:9083 \
--conf hive.server2.thrift.bind.host=192.168.72.18 \
--conf spark.executor.heartbeatInterval=43200s \
--conf spark.storage.blockManagerSlaveTimeoutMs=43200s \
--conf spark.blacklist.enabled=true \
--conf spark.sql.adaptive.enabled=true \
--conf spark.scheduler.listenerbus.eventqueue.capacity=100000 \
--conf spark.network.timeout=43300s \
--conf spark.akka.timeout=43200s \
--conf spark.pyspark.python=Python/bin/python3 \
--conf spark.pyspark.driver.python=Python/bin/python3 \
--archives=hdfs://ns-hf/project/aegis/cjzhu3/PythonAarch64.zip#Python \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--queue compass \
./test.py
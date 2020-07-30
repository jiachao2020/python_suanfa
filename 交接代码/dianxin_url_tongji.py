#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import ArrayType, StringType

sc = SparkContext(appName="dianxin_chushu_yugu").getOrCreate()
sqlContext = SQLContext(sc)
data = sqlContext.read.text("/user/chenzhixiang/dianxin_url_group.txt")
# data = data.withColumn('value', data['value'].cast(ArrayType(StringType())))
host_list = data.collect()
df_all = sqlContext.sql('SELECT upper(mobile) AS mobile,lower(host) as host FROM dpi.dpi_origin')
df_all_dis = df_all.distinct()
df_all_dis.persist()
# df_all_dis.persist(storageLevel='MEMORY_AND_DISK_SER')
new_mobile = sqlContext.createDataFrame([('000000599B2A34840079E324B67534B8',)], ['mobile'])
for i in range(1):
    host_group = host_list[i]['value']
    host_group_list = host_group.strip().split(', ')
    print(host_group_list, type(host_group_list))
    df_mobile_host_num = df_all_dis.filter(df_all_dis.host.isin(host_group_list)).groupBy('mobile').count()
    # df_mobile_host_num.show()
    df_mobile = df_mobile_host_num.filter('count={}'.format(len(host_group_list))).drop('count')
    print(len(host_group_list))
    # df_mobile.show()
    new_mobile = new_mobile.union(df_mobile)
new_mobile.repartition(1).write.format('text').saveAsTable('rac.dianxin_jiaotong_chika_20190114')
sc.stop()

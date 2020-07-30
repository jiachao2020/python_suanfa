#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType

sc = SparkContext(appName="dianxin_TGI").getOrCreate()
sqlContext = SQLContext(sc)
dianxin_all_host_num = '''
SELECT lower(HOST) as host,
       count(DISTINCT upper(mobile)) AS people_num
FROM dpi.dpi_origin
GROUP BY lower(HOST)
'''
dianxin_all_host_goodsample_num = '''
SELECT lower(a.host) as host,
       count(DISTINCT upper(a.mobile)) AS good_sample_num
FROM dpi.dpi_origin AS a
LEFT JOIN chenzhixiang.pufa_good_sample AS b ON upper(a.mobile)=b.mobile
WHERE b.mobile IS NOT NULL
GROUP BY lower(a.host)
'''
df_dianxin_all_host_num = sqlContext.sql(dianxin_all_host_num)

df_dianxin_all_host_goodsample_num = sqlContext.sql(dianxin_all_host_goodsample_num)
df_tgi_num = df_dianxin_all_host_num.join(df_dianxin_all_host_goodsample_num, on='host', how='left').drop(
    df_dianxin_all_host_goodsample_num.host).na.fill(0)
df_tgi_num = df_tgi_num.withColumn('good_sample_num', df_tgi_num['good_sample_num'].cast(IntegerType()))
df_tgi_num = df_tgi_num.withColumn('people_num', df_tgi_num['people_num'].cast(IntegerType()))
dianxin_all_host_all_num = '''
SELECT count(DISTINCT upper(mobile)) AS all_people_num
FROM dpi.dpi_origin
'''
df_dianxin_all_host_all_num = sqlContext.sql(dianxin_all_host_all_num)
all_num_value = int(df_dianxin_all_host_all_num.collect()[0]['all_people_num'])

dianxin_all_host_all_goodsample_num = '''SELECT count(DISTINCT upper(a.mobile)) AS all_good_sample_num
FROM dpi.dpi_origin AS a
LEFT JOIN chenzhixiang.pufa_good_sample AS b ON upper(a.mobile)=upper(b.mobile)
WHERE b.mobile IS NOT NULL
'''
df_dianxin_all_host_all_goodsample_num = sqlContext.sql(dianxin_all_host_all_goodsample_num)
all_goodsample_value = int(df_dianxin_all_host_all_goodsample_num.collect()[0]['all_good_sample_num'])

df_tgi_num.select('host', 'good_sample_num', 'people_num',
                  ((df_tgi_num.good_sample_num / all_goodsample_value) / (
                          df_tgi_num.people_num / all_num_value)).alias('rate')).sort('rate',
                                                                                      ascending=False).repartition(
    1).write.csv(
    os.path.join(
        '/user/chenzhixiang', 'credit_dianxin_tgi'), mode='overwrite', header=True)

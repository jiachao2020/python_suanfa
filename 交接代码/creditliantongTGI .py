#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType

sc = SparkContext(appName="liantong_TGI").getOrCreate()
sqlContext = SQLContext(sc)
liantong_all_host_num = '''
SELECT HOST,
       count(DISTINCT upper(mobile)) AS people_num
FROM dpi.dpi_origin_931
GROUP BY HOST
'''
liantong_all_host_goodsample_num = '''
SELECT a.host,
       count(DISTINCT upper(a.mobile)) AS good_sample_num
FROM dpi.dpi_origin_931 AS a
LEFT JOIN chenzhixiang.pufa_good_sample AS b ON upper(a.mobile)=b.mobile
WHERE b.mobile IS NOT NULL
GROUP BY a.host
'''
df_liantong_all_host_num = sqlContext.sql(liantong_all_host_num)

df_liantong_all_host_goodsample_num = sqlContext.sql(liantong_all_host_goodsample_num)
df_tgi_num = df_liantong_all_host_num.join(df_liantong_all_host_goodsample_num, on='host', how='left').drop(
    df_liantong_all_host_goodsample_num.host).na.fill(0)
df_tgi_num = df_tgi_num.withColumn('good_sample_num', df_tgi_num['good_sample_num'].cast(IntegerType()))
df_tgi_num = df_tgi_num.withColumn('people_num', df_tgi_num['people_num'].cast(IntegerType()))
liantong_all_host_all_num = '''
SELECT count(DISTINCT upper(mobile)) AS all_people_num
FROM dpi.dpi_origin_931
'''
df_liantong_all_host_all_num = sqlContext.sql(liantong_all_host_all_num)
all_num_value = int(df_liantong_all_host_all_num.collect()[0]['all_people_num'])

liantong_all_host_all_goodsample_num = '''SELECT count(DISTINCT upper(a.mobile)) AS all_good_sample_num
FROM dpi.dpi_origin_931 AS a
LEFT JOIN chenzhixiang.pufa_good_sample AS b ON upper(a.mobile)=upper(b.mobile)
WHERE b.mobile IS NOT NULL
'''
df_liantong_all_host_all_goodsample_num = sqlContext.sql(liantong_all_host_all_goodsample_num)
all_goodsample_value = int(df_liantong_all_host_all_goodsample_num.collect()[0]['all_good_sample_num'])

df_tgi_num.select('host', 'good_sample_num', 'people_num',
                  ((df_tgi_num.good_sample_num / all_goodsample_value) / (
                          df_tgi_num.people_num / all_num_value)).alias('rate')).sort('rate',
                                                                                      ascending=False).repartition(
    1).write.csv(
    os.path.join(
        '/user/chenzhixiang', 'credit_liantong_tgi'), mode='overwrite', header=True)

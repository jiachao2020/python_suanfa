#!/usr/bin/env python
# -*- coding:utf-8 -*-

# unicode setting for chinese characters

import os

# pyspark libs
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, rand, randn, md5
from pyspark.sql.types import StringType, StructField, StructType, json, IntegerType

if __name__ == "__main__":

    sc = SparkContext(appName='qianzai_sample_split').getOrCreate()
    sqlContext = SQLContext(sc)
    # 读取正样本和潜在样本的label
    final_2016 = sc.textFile('/user/hive/warehouse/rac.db/liantong_huanbei_mobile_label_20181210/0*')
    # 读取联通所有数据的tf_idf的特征值
    # x = sc.textFile("/user/hive/warehouse/rac.db/liantong_x3_20181017/0*")
    final_2016 = final_2016.map(lambda row: row.strip().split('\001'))  # python 3
    final_2016 = final_2016.toDF(["mobile", "label"])

    df_good_sample = final_2016.filter(final_2016['label'] == 1)
    df_qianzai_sample = final_2016.filter(final_2016['label'] == 0)
    weight_list = [1 / 13] * 13
    df_splits = df_qianzai_sample.randomSplit(weights=weight_list, seed=1)

    # x = x.map(lambda row: row.strip().split(','))
    # x = x.toDF([str(i) for i in range(790)]+['mobile'])
    # final_2016 = final_2016.withColumn("mobile",final_2016["mobile"].cast(StringType()))
    # final_2016 = final_2016.withColumn("label",final_2016["label"].cast(IntegerType()))
    for i in range(1):
        # df_y = df_splits[i].unionAll(df_good_sample)
        df_splits[i].unionAll(df_good_sample).repartition(1).write.csv(
            os.path.join("/user/chenzhixiang/tfidf_huanbei_20181210/huanbei_tfidf{}".format(i), 'y'), header=True)
        # x_inner = x.join(df_y, x.mobile == df_y.mobile, how='inner').drop([df_y.mobile, df_y.label])
        # x.repartition(1).write.csv(os.path.join("/user/chenzhixiang/liantong_tfidf{}".format(3), 'x_inner'), header=True)

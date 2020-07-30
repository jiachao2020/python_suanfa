#!/usr/bin/python
# coding: utf-8
# encoding: utf-8

import os
import sys

from pyspark import SparkConf, SparkContext
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SQLContext
from pyspark.sql import Row, functions as F
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import desc
from pyspark.sql.types import *
from pyspark.sql.window import Window

sc = SparkContext().getOrCreate()
sqlContext = SQLContext(sc)

lines = sc.textFile("/user/hive/warehouse/public.db/game_chuanqi_als_result_20181018/0*")
parts = lines.map(lambda row: row.split("\001"))
parts = parts.toDF(["user", "item", "rating"])

p_item = parts.select("item").distinct().withColumn("itemId", F.row_number().over(Window.orderBy("item")))

p_item = p_item.withColumnRenamed("item", "item1")
parts_1 = parts.join(p_item, p_item["item1"] == parts["item"], "left_outer")
parts_1 = parts_1.toDF("user", "item", "rating", "item1", "itemId")
baoming = parts_1.select("itemId").where("item=='zheng'").distinct()

baoming = baoming.rdd.map(lambda x: x[0]).take(1)
baoming = baoming[0]
# print(baoming)

p_user = parts.select("user").distinct().withColumn("userId", F.row_number().over(Window.orderBy("user")))
num_user = p_user.count()
p_user = p_user.withColumnRenamed("user", "user1")
parts_2 = parts_1.join(p_user, p_user["user1"] == parts_1["user"])
# p_user.repartition(1).write.csv(os.path.join("/user/xuqian/qiche/ALS/user_label"),mode="overwrite")
parts = parts_2.select(parts_2["userId"], parts_2["itemId"], parts_2["rating"])

parts = parts.withColumn("rating", parts["rating"].cast(FloatType()))
# (training, test) = parts.randomSplit([0.8, 0.2])

training = parts
test = parts
# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(rank=200, maxIter=100, regParam=0.1, numUserBlocks=2000, numItemBlocks=2000, implicitPrefs=False, alpha=1.0,
          userCol='userId',
          itemCol='itemId', seed=None, ratingCol='rating', nonnegative=False, checkpointInterval=1000,
          intermediateStorageLevel='MEMORY_AND_DISK',
          finalStorageLevel='MEMORY_AND_DISK', coldStartStrategy='drop')

model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
# print("Root-mean-square error = " + str(rmse))

item_rec = parts.select(als.getItemCol()).distinct()
item_rec = item_rec.where(item_rec.itemId == baoming)
item_rec = model.recommendForItemSubset(item_rec, num_user)

data = item_rec.select("recommendations")
data_list = data.rdd.map(lambda x: x[0]).take(1)
data = [data_list[0][i] for i in range(len(data_list[0]))]
data1 = sqlContext.createDataFrame(data)

p_user = p_user.withColumnRenamed("userId", "userId_1")

data2 = data1.join(p_user, data1["userId"] == p_user["userId_1"])
data3 = data2.select("user1", "rating")
data3 = data3.sort("rating", ascending=False)

data3 = data3.withColumn("rn", F.row_number().over(Window.orderBy("user1")))
data3 = data3.where(data3["rn"] <= 150000).drop("rn", "rating")

data3.repartition(1).write.csv(os.path.join("/user/maojiashun/", 'ALS_ruipeng'), mode="overwrite")

print("Root-mean-square error = " + str(rmse))

sc.stop()

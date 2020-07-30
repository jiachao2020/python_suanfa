#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ml  program for any alg except dl.注意：spark 2.0以后的版本需要用VectorAssembler将特征进行组合生成一个向量features，
# ParamGridBuilder中添加的网格必须是一个数组，并且有多个元素

from __future__ import print_function

import os

import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Word2Vec
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.util import MLWritable
from pyspark.sql import HiveContext
from pyspark.sql import SQLContext, Row
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, isnull
from pyspark.sql.types import IntegerType, DoubleType
from sklearn.metrics import roc_curve, auc

if __name__ == "__main__":
    sc = SparkContext().getOrCreate()
    sqlContext = SQLContext(sc)

    # get input data ."mobile","label",var_name_list
    ori_data = sc.textFile("/user/wangkang/qiche/word2vec/word2vec_data/p*")
    ori_data = ori_data.map(lambda row: row.strip().split(',')).toDF(["mobile", "label", "sentence"])
    ori_data = ori_data.select("mobile", "label", "sentence").rdd.map(
        lambda x: Row(mobile=int(x['mobile']), label=int(x['label']), sentence=(x['sentence'].split(" ")))).toDF()
    word2Vec = Word2Vec(vectorSize=400, minCount=1, numPartitions=1, stepSize=0.1, maxIter=10, seed=1, windowSize=5,
                        maxSentenceLength=100, inputCol="sentence", outputCol="result")
    model = word2Vec.fit(ori_data)
    predict_result = model.transform(ori_data)
    predict_result = predict_result.withColumn("result", predict_result["result"].cast(StringType()))
    # predict_result.select("mobile","label","result").rdd.map(lambda x: [x.strip().split(',"')[0].split(',')[0]]+[x.strip().split(',"')[0].split(',')[1]]+x.strip().split(',"')[1].strip('[').strip(']"').split(',')).toDF()
    predict_result.select("mobile", "label", "result").repartition(1).write.csv(
        '/user/wangkang/qiche/word2vec/word2Vec_vec', mode='overwrite')
    predict_result = sc.textFile('/user/wangkang/qiche/word2vec/word2Vec_vec/p*')
    predict_result = predict_result.map(
        lambda x: [x.strip().split(',"')[0].split(',')[0]] + [x.strip().split(',"')[0].split(',')[1]] +
                  x.strip().split(',"')[1].strip('[').strip(']"').split(',')).toDF()
    predict_result.repartition(1).write.csv('/user/wangkang/qiche/word2vec/word2Vec_vec_2', mode='overwrite')

    # Kmeans模型
    ori_data = sc.textFile("/user/wangkang/qiche/word2vec/word2Vec_vec_2/p*")
    ori_data = ori_data.map(lambda row: row.strip().split(','))
    ori_data = ori_data.toDF(['mobile'] + ['label'] + [str(i) for i in range(400)])

    data = ori_data.withColumn("label", ori_data["label"].cast(IntegerType()))
    data.persist()
    data = data.dropDuplicates()
    data = data

    col_name = [str(i) for i in range(400)]
    for i in col_name:
        data = data.withColumn(i, data[i].cast(DoubleType()))
        dataWithFeatures = VectorAssembler(inputCols=col_name, outputCol="features")

    data = dataWithFeatures.transform(data)

    trainingData = data
    testData = data
    '''RFC = RandomForestClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability", rawPredictionCol="rawPrediction",
                                maxDepth=14, maxBins=32, minInstancesPerNode=2, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=True, checkpointInterval=10, 
                                impurity="gini", numTrees=2000, featureSubsetStrategy="auto", seed=None, subsamplingRate=0.8)'''
    BKM = BisectingKMeans(featuresCol="features", predictionCol="prediction", maxIter=20, seed=1, k=4,
                          minDivisibleClusterSize=1.0)
    # Train model.
    # paramMap1 = {gbt.stepSize: 0.05,gbt.minInstancesPerNode:2,gbt.maxDepth:6,gbt.cacheNodeIds:True,gbt.subsamplingRate:1,gbt.maxIter:200}
    BKM_model = BKM.fit(trainingData)
    cost = BKM_model.computeCost(trainingData)
    print("Within Set Sum of Squared Errors = " + str(cost))
    centers = BKM_model.clusterCenters()
    summary = BKM_model.summary
    print(summary.k)
    print(summary.clusterSizes)

    # Make predictions. #"features", "label","mobile","prediction"
    prediction_result = BKM_model.transform(testData)
    # prediction_result.take(5)
    prediction_result.select('mobile', 'label', 'prediction').repartition(1).write.csv(
        '/user/wangkang/qiche/word2vec/Kmeans', mode='overwrite')
'''conf = SparkConf()
conf.setAppName('a')
sc = SparkContext(conf=conf)
'''
hc = HiveContext(sc)
data = sc.textFile('/user/wangkang/qiche/word2vec/Kmeans/p*')
data = data.map(lambda row: row.strip().split(',')).toDF(["mobile", "label", "prediction"])
kmeans_mobile = data.createOrReplaceTempView('kmeans_mobile')
load_all = "select distinct c.mobile as mobile,lower(d.uid) as uid,'kmeans' as rule_name,'0' as operator,d.city_code as city_code from \
               (select b.mobile,a.prediction from \
               (select prediction,label,case when prediction='0' and label='1' then count(distinct mobile) \
                when prediction='1' and label='1' then count(distinct mobile) \
                when prediction='2' and label='1' then count(distinct mobile) \
               when prediction='3' and label='1' then count(distinct mobile) end number from kmeans_mobile \
              group by prediction,label order by number desc limit 1 ) a left join kmeans_mobile b\
                on a.prediction=b.prediction) c left join bigdata_yunchetong.e_666_dpi_data d on c.mobile=d.mobile  where d.p_class='http'"
load_all = hc.sql(load_all)
load_all.repartition(1).write.csv("/user/wangkang/data/send_data/20181114", mode="append")

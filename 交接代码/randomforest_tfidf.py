#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ml  program for any alg except dl.注意：spark 2.0以后的版本需要用VectorAssembler将特征进行组合生成一个向量features，
# ParamGridBuilder中添加的网格必须是一个数组，并且有多个元素

from __future__ import print_function

import math
import os

import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
# from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.util import MLWritable
from pyspark.sql import Row
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, isnull
from pyspark.sql.types import IntegerType, DoubleType, StringType, FloatType
from sklearn.metrics import roc_curve, auc

if __name__ == "__main__":
    sc = SparkContext(appName='RandomForest').getOrCreate()
    sqlContext = SQLContext(sc)

    # get input data ."mobile","label",var_name_list
    ori_data = sc.textFile("/user/hive/warehouse/rac.db/liantong_host_xy0_20181026/0*")
    ori_data = ori_data.map(lambda row: row.strip().split(','))
    ori_data = ori_data.toDF([str(i) for i in range(1270)] + ["mobile", "label"])

    data = ori_data.withColumn("label", ori_data["label"].cast(IntegerType()))
    data = data.filter(data['label'] == 1).limit(100).unionAll(data.filter(data['label'] == 0).limit(100))
    data.persist()
    data = data.dropDuplicates()

    col_name = [str(i) for i in range(1270)]
    for i in col_name:
        data = data.withColumn(i, data[i].cast(DoubleType()))

    # save mobile and x for other use
    data_x = data.select(["mobile"] + col_name)
    # data_x.persist()
    data_x.repartition(1).write.csv(os.path.join("/user/chenzhixiang/tfidf_host/liantong_tfidf0/train", 'x'),
                                    mode='overwrite', header=True)

    # save mobile and y for other use
    data_y = data.select(["mobile", "label"])
    # data_y.persist()
    data_y.repartition(1).write.csv(os.path.join("/user/chenzhixiang/tfidf_host/liantong_tfidf0/train", 'y'),
                                    mode='overwrite', header=True)

    # 进入后续模型训练,gbt require vector for features format
    # "features", "label","mobile"
    dataWithFeatures = VectorAssembler(inputCols=col_name, outputCol="features")
    data = dataWithFeatures.transform(data)

    trainingData = data
    testData = data
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", predictionCol="prediction",
                                probabilityCol="probability", rawPredictionCol="rawPrediction",
                                maxDepth=14, maxBins=32, minInstancesPerNode=2, minInfoGain=0.0, maxMemoryInMB=256,
                                cacheNodeIds=True, checkpointInterval=10,
                                impurity="gini", numTrees=2000, featureSubsetStrategy="auto", seed=None,
                                subsamplingRate=0.8)

    # Train model.
    # paramMap1 = {gbt.stepSize: 0.1, gbt.minInstancesPerNode: 30, gbt.maxDepth: 6, gbt.cacheNodeIds: True,
    #              gbt.subsamplingRate: 1, gbt.maxIter: 20}
    rf_model = rf.fit(trainingData)

    # Make predictions. "features", "label","mobile","prediction"
    prediction_result = rf_model.transform(testData)
    print("featureImportances is: ")
    print(rf_model.featureImportances)

    # save model
    modelPath = "/user/chenzhixiang/tfidf_host/liantong_tfidf0/rf_model"
    gbt_model.save(modelPath)  # load

    # get metrics
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction",
                                              metricName="areaUnderROC")  # areaUnderPR  areaUnderROC
    accuracy_all = evaluator.evaluate(prediction_result)
    print("areaUnderROC ALL  = %g" % (accuracy_all))

    # save predict_result
    prediction_result = prediction_result.select("probability", "prediction", "label", "mobile").rdd.map(
        lambda x: Row(probability=str(x['probability']), prediction=int(x['prediction']), label=int(x['label']),
                      mobile=str(x['mobile']))).toDF()
    # prediction_result_format = prediction_result.withColumn("probability",
    #                                                         prediction_result["probability"].cast(StringType())).select(
    #     "probability", "prediction", "label", "mobile").repartition(1)
    # prediction_result = prediction_result.select("probability", "prediction", "label", "mobile").repartition(1)
    prediction_result.repartition(1).write.csv(
        os.path.join("/user/chenzhixiang/tfidf_host/liantong_tfidf0", 'prediction_result'), mode='overwrite',
        header=False)
    predict_result = sc.textFile('/user/chenzhixiang/tfidf_host/liantong_tfidf0/prediction_result/p*')
    predict_result = predict_result.map(
        lambda x: [x.strip().split(',')[0]] + [x.strip().split(',')[1]] + [x.strip().split(',')[2]] +
                  [x.strip().split(',')[3].strip('"').strip('[').strip(']').split(',')[0]] + [
                      x.strip().split(',')[4].strip('"').strip('"').strip(']').strip('\\')]).toDF(
        ['label', 'mobile', 'predict', 'neg_probability', 'pos_probability'])
    predict_result.repartition(1).write.csv(
        os.path.join("/user/chenzhixiang/tfidf_host/liantong_tfidf0", 'predict_result'), mode='overwrite', header=True)
    '''cols = ['label', 'prediction', 'probability']
    prediction_result_topd = prediction_result.select('label', 'prediction', 'probability').toPandas()
    m = ModelResultAnalysis(prediction_result_topd, cols).cal_ks()'''
    predict_result = predict_result.withColumn('label', predict_result['label'].cast(IntegerType()))
    predict_result = predict_result.withColumn('predict', predict_result['predict'].cast(IntegerType()))
    predict_result = predict_result.withColumn('pos_probability', predict_result['pos_probability'].cast(FloatType()))
    prediction_result_topd = predict_result.toPandas()
    prediction_result_topd = pd.DataFrame(prediction_result_topd, columns=['label', 'predict', 'pos_probability'])
    print(prediction_result_topd.head())

    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, \
        roc_auc_score

    # y_true = prediction_result_topd['label']
    # y_pred = prediction_result_topd['probability']
    # fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    # ks_ary = tpr - fpr
    # ks = np.max(ks_ary)
    # acc = accuracy_score(y_true, y_pred)
    # avg_precision = average_precision_score(y_true, y_score)
    # pre = precision_score(y_true, y_pred)
    # re = recall_score(y_true, y_pred)
    # auc = roc_auc_score(y_true, y_score)
    # f1 = f1_score(y_true, y_pred)
    # print(m, acc, avg_precision, pre, re, auc, ks, f1)
    fpr, tpr, threshold = roc_curve(prediction_result_topd['label'], prediction_result_topd['pos_probability'],
                                    pos_label=1)
    print(fpr)
    ks_ary = tpr - fpr
    ks = np.max(ks_ary)
    print(ks)
    max_ks_threshold = np.argmax(tpr - fpr)
    print(threshold[max_ks_threshold])
    prediction_result_topd = prediction_result_topd.sort_values(by=['pos_probability'], axis=0, ascending=False)
    prediction_result = prediction_result_topd[
        (prediction_result_topd['pos_probability'] >= threshold[max_ks_threshold]) & (
                prediction_result_topd['label'] == 0)]
    new_prediction_result = sqlContext.createDataFrame(prediction_result)
    new_prediction_result.repartition(1).write.csv(
        os.path.join("/user/chenzhixiang/tfidf_host/liantong_tfidf0", 'result'), mode='overwrite',
        header=False)
    print(ks)
    y_true = prediction_result_topd['label']
    y_pred = prediction_result_topd['pos_probability']
    acc = accuracy_score(y_true, y_pred.round())
    avg_precision = average_precision_score(y_true, y_pred.round())
    pre = precision_score(y_true, y_pred.round(), pos_label=1)
    re = recall_score(y_true, y_pred.round(), pos_label=1)
    auc = roc_auc_score(y_true, y_pred.round())
    # print(auc)
    f1 = f1_score(y_true, y_pred.round(), pos_label=1)
    print(acc, avg_precision, pre, re, auc, f1)

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

# class ModelResultAnalysis(object):
#     def __init__(self, X_train, columns=["real", "predict", "Score"], num_split=20):
#         """
#         X_train: 输入数据，目前只支持pandas.dataframe
#         columns : 输入列的顺序是固定的，排列为真实标签，预测标签，预测为bad的分数
#         num_split: 结果分析划分的比例，默认是20
#         """
#         self.X_train = X_train[columns]
#         self.shape0 = self.X_train.shape[0]
#         self.columns = columns
#         self.num = num_split
#
#     def sort_train(self, var="Score"):
#         return self.X_train.sort_values(by=var, ascending=False)
#
#     def split_train(self, n, n_part=10):
#         if n > n_part:
#             p = int(n / n_part)  # 步长，每个bin区间样本
#             if p * n_part < n:
#                 return [x for x in range(0, p * n_part, p)]  # 获得每个bin区间的Index
#             else:  # ==
#                 return [x for x in range(0, n, p)]
#         else:
#             return None
#
#     def cal_ks(self, bad_label=1):
#         data_dict = []
#         begin_index = 0
#         bad = 0
#         cnt = 0
#         sort_var = self.columns[2]
#         print('sort_var:', sort_var)
#         st = self.sort_train(var=sort_var)  # 保存按照分数排序后的样本
#         te = self.split_train(self.shape0, n_part=self.num)  # 保存每个bin区间的首index
#         if te is None:
#             print('cal_ks:bad te:', te)
#             return None
#
#         n = len(te)
#         ii = 0
#         cum_sample_pro = 0
#         while ii < n:
#             if ii + 1 <= n - 1:  # 还不到最后一个bin区间
#                 test = st.iloc[te[ii]:te[ii + 1]]
#                 sampleSize = te[ii + 1] - te[ii]
#                 sp = sampleSize / self.shape0  # 当前bin区间样本数占总样本比率
#
#             else:
#                 test = st.iloc[te[ii]:]
#                 sampleSize = self.shape0 - te[ii]
#                 sp = sampleSize / self.shape0
#
#             cum_sample_pro += sp
#
#             MinScore = int(1000 * test[sort_var].min())
#             MeanScore = int(1000 * test[sort_var].mean())
#             MaxScore = int.ceil(1000 * test[sort_var].max())
#             badRate = round((test[self.columns[0]] == bad_label).astype("int").mean(), 4)
#             bad = bad + badRate * test.shape[0]
#             cnt = cnt + test.shape[0]
#             CumBad = round(bad / cnt, 4)
#             # use round 2
#             data_dict += [round(cum_sample_pro, 2) * 100, sampleSize, MinScore, MeanScore, MaxScore, badRate,
#                           CumBad]  # 计算每个Bin中的值
#             ii += 1
#
#         data = pd.DataFrame(np.array(data_dict).reshape(self.num, 7),
#                             columns=["CumProp", "sampleSize", "MinScore", "MeanScore", "MaxScore", "BadRate", "CumBad"])
#         data["BadCap"] = ((data.BadRate * data.sampleSize / sum(data.BadRate * data.sampleSize)).cumsum()).round(4)
#         data["GoodCap"] = (
#             ((1 - data.BadRate) * data.sampleSize / sum((1 - data.BadRate) * data.sampleSize)).cumsum()).round(4)
#         data["Ks"] = data["BadCap"] - data["GoodCap"]
#         return data

if __name__ == "__main__":
    sc = SparkContext(appName='GBDT').getOrCreate()
    sqlContext = SQLContext(sc)
    riqi = '20181207'
    # get input data ."mobile","label",var_name_list
    ori_data = sc.textFile("/user/hive/warehouse/rac.db/liantong_huanbei_xy0_{}/0*".format(riqi))
    ori_data = ori_data.map(lambda row: row.strip().split(','))
    ori_data = ori_data.toDF([str(i) for i in range(925)] + ["mobile", "label"])

    data = ori_data.withColumn("label", ori_data["label"].cast(IntegerType()))
    # data = data.filter(data['label'] == 1).limit(100).unionAll(data.filter(data['label'] == 0).limit(100))
    data.persist()
    data = data.dropDuplicates()

    col_name = [str(i) for i in range(925)]
    # for i in col_name:
    #     data = data.withColumn(i, data[i].cast(DoubleType()))
    data = data.select([data[str(i)].cast(DoubleType()) for i in range(925)] + [data["mobile"], data['label']])
    # save mobile and x for other use
    data_x = data.select(["mobile"] + col_name)
    # data_x.persist()
    data_x.repartition(1).write.csv(
        os.path.join("/user/chenzhixiang/tfidf_huanbei_{}/huanbei_tfidf0/train".format(riqi), 'x'),
        mode='overwrite', header=True)

    # save mobile and y for other use
    data_y = data.select(["mobile", "label"])
    # data_y.persist()
    data_y.repartition(1).write.csv(
        os.path.join("/user/chenzhixiang/tfidf_huanbei_{}/huanbei_tfidf0/train".format(riqi), 'y'),
        mode='overwrite', header=True)

    # 进入后续模型训练,gbt require vector for features format
    # "features", "label","mobile"
    dataWithFeatures = VectorAssembler(inputCols=col_name, outputCol="features")
    data = dataWithFeatures.transform(data)

    trainingData = data
    testData = data
    gbt = GBTClassifier(labelCol="label", featuresCol="features", predictionCol="prediction", maxBins=32, maxIter=200,
                        maxDepth=6
                        , stepSize=0.05, minInstancesPerNode=2, minInfoGain=0.0, seed=1)

    # Train model.
    # paramMap1 = {gbt.stepSize: 0.1, gbt.minInstancesPerNode: 30, gbt.maxDepth: 6, gbt.cacheNodeIds: True,
    #              gbt.subsamplingRate: 1, gbt.maxIter: 20}
    gbt_model = gbt.fit(trainingData)

    # Make predictions. "features", "label","mobile","prediction"
    prediction_result = gbt_model.transform(testData)
    print("featureImportances is: ")
    print(gbt_model.featureImportances)

    # save model
    modelPath = "/user/chenzhixiang/tfidf_huanbei_{}/huanbei_tfidf0/gbt_model"
    gbt_model.save(modelPath.format(riqi))  # load

    # get metrics
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction",
                                              metricName="areaUnderROC")  # areaUnderPR  areaUnderROC
    accuracy_all = evaluator.evaluate(prediction_result)
    print("areaUnderROC ALL  = %g" % (accuracy_all))

    # save predict_result
    prediction_result = prediction_result.select("probability", "prediction", "label", "mobile").rdd.map(
        lambda x: Row(probability=str(x['probability']), prediction=int(x['prediction']), label=int(x['label']),
                      mobile=str(x['mobile']))).toDF()
    prediction_result.repartition(1).write.csv(
        os.path.join("/user/chenzhixiang/tfidf_huanbei_{}/huanbei_tfidf0".format(riqi), 'prediction_result'),
        mode='overwrite',
        header=False)
    predict_result = sc.textFile('/user/chenzhixiang/tfidf_huanbei_{}/huanbei_tfidf0/prediction_result/p*'.format(riqi))
    predict_result = predict_result.map(
        lambda x: [x.strip().split(',')[0]] + [x.strip().split(',')[1]] + [x.strip().split(',')[2]] +
                  [x.strip().split(',')[3].strip('"').strip('[').strip(']').split(',')[0]] + [
                      x.strip().split(',')[4].strip('"').strip('"').strip(']').strip('\\')]).toDF(
        ['label', 'mobile', 'predict', 'neg_probability', 'pos_probability'])
    predict_result.repartition(1).write.csv(
        os.path.join('/user/chenzhixiang/tfidf_huanbei_{}/huanbei_tfidf0'.format(riqi), 'predict_result'),
        mode='overwrite',
        header=True)

    # cols = ['label', 'prediction', 'probability']
    # prediction_result_topd = prediction_result.select('label', 'prediction', 'probability').toPandas()
    # m = ModelResultAnalysis(prediction_result_topd, cols).cal_ks()

    predict_result = predict_result.withColumn('label', predict_result['label'].cast(IntegerType()))
    predict_result = predict_result.withColumn('predict', predict_result['predict'].cast(IntegerType()))
    predict_result = predict_result.withColumn('pos_probability', predict_result['pos_probability'].cast(FloatType()))
    prediction_result_topd = predict_result.toPandas()
    prediction_result_topd = pd.DataFrame(prediction_result_topd,
                                          columns=['mobile', 'label', 'predict', 'pos_probability'])
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
    max_ks_threshold = np.argmax(ks_ary)
    print(threshold[max_ks_threshold])
    prediction_result_topd = prediction_result_topd.sort_values(by=['pos_probability'], axis=0, ascending=False)
    prediction_result = prediction_result_topd[
        (prediction_result_topd['pos_probability'] >= threshold[max_ks_threshold]) & (
                prediction_result_topd['label'] == 0)]['mobile'].to_frame()
    print(type(prediction_result), prediction_result.head())
    new_prediction_result = sqlContext.createDataFrame(prediction_result)
    new_prediction_result.repartition(1).write.csv(
        os.path.join("/user/chenzhixiang/tfidf_huanbei_{}/huanbei_tfidf0".format(riqi), 'result'), mode='overwrite',
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

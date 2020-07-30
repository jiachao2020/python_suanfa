#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

import numpy as np
import pandas as pd
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType, DoubleType, FloatType
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, \
    recall_score, roc_auc_score, fbeta_score, roc_curve

riqi = '20190202'
spark = SparkSession.builder.appName("zx_classifier").getOrCreate()

dataset = spark.read.table('rac.liantong_zx_mobile_label_vec_20181227')
dataset = dataset.rdd.map(lambda x: [x['mobile']] + [x['label']] + x['model'].strip('[').strip(']').split(',')).toDF(
    ['mobile'] + ['label'] + [str(i) for i in range(200)])
dataset = dataset.withColumn('label', dataset['label'].cast(IntegerType()))
dataset = dataset.select(
    [dataset['mobile'], dataset['label']] + [dataset[str(i)].cast(DoubleType()) for i in range(200)])
# set seed for reproducibility
trainingData, testData = dataset.randomSplit([0.8, 0.2], seed=100)

ignore = ['mobile', 'label']
assembler = VectorAssembler(
    inputCols=[x for x in trainingData.columns if x not in ignore],
    outputCol='features')

train_data = assembler.transform(trainingData).select('mobile', 'label', 'features')
# with 250 iterations, GINI is around ~0.276 for submission
gbt = GBTClassifier(labelCol='label', featuresCol='features', predictionCol="prediction", maxBins=32,
                    lossType='logistic', maxIter=300, maxDepth=6, stepSize=0.03, minInstancesPerNode=20,
                    minInfoGain=0, maxMemoryInMB=1024, seed=1, subsamplingRate=1)

evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label', metricName='areaUnderROC')

# no parameter search
paramGrid = ParamGridBuilder().build()

# 6-fold cross validation
# cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=2, parallelism=700)
# cvModel = cv.fit(train_data)

tvs = TrainValidationSplit(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8,
                           parallelism=200, seed=1)
tvsModel = tvs.fit(train_data)

print("trained GBT classifier:%s" % tvsModel)
print("featureImportances is: ")
print(tvsModel.bestModel.featureImportances)

# display CV score
auc_roc = tvsModel.validationMetrics[0]
print("AUC ROC = %g" % auc_roc)
gini = (2 * auc_roc - 1)
print("GINI ~=%g" % gini)

# save model
modelPath = "/user/chenzhixiang/zx_class_{}/tvsModel"
tvsModel.save(modelPath.format(riqi))

# error_array = tvsModel.bestModel.evaluateEachIteration(train_data)
# spark.createDataFrame(error_array).repartition(1).write.csv('/user/chenzhixiang/zx_class_{}/error_array'.format(riqi),
#                                                             header=False)
# tvsModel1 = TrainValidationSplitModel.load(modelPath.format(riqi))

# prepare submission
prediction_result = tvsModel.transform(assembler.transform(testData).select('mobile', 'label', 'features'))
print(evaluator.evaluate(prediction_result))
spark.createDataFrame([(auc_roc, gini, evaluator.evaluate(prediction_result))],
                      ['auc_roc', 'gini', 'evaluator']).repartition(1).write.csv(
    '/user/chenzhixiang/zx_class_{}/validation_metrics'.format(riqi), header=True)

prediction_result = prediction_result.select("probability", "prediction", "label", "mobile").rdd.map(
    lambda x: Row(probability=str(x['probability']), prediction=int(x['prediction']), label=int(x['label']),
                  mobile=str(x['mobile']))).toDF()
prediction_result.repartition(1).write.csv(
    os.path.join("/user/chenzhixiang/zx_class_{}/".format(riqi), 'prediction_result'), mode='overwrite', header=False)
predict_result = spark.sparkContext.textFile('/user/chenzhixiang/zx_class_{}/prediction_result/p*'.format(riqi))
predict_result = predict_result.map(
    lambda x: [x.strip().split(',')[0]] + [x.strip().split(',')[1]] + [x.strip().split(',')[2]] +
              [x.strip().split(',')[3].strip('"').strip('[').strip(']').split(',')[0]] + [
                  x.strip().split(',')[4].strip('"').strip('"').strip(']').strip('\\')]).toDF(
    ['label', 'mobile', 'predict', 'neg_probability', 'pos_probability'])
predict_result.repartition(1).write.csv(
    os.path.join('/user/chenzhixiang/zx_class_{}/'.format(riqi), 'predict_result'),
    mode='overwrite',
    header=True)
# change df type
predict_result = predict_result.withColumn('label', predict_result['label'].cast(IntegerType()))
predict_result = predict_result.withColumn('predict', predict_result['predict'].cast(IntegerType()))
predict_result = predict_result.withColumn('pos_probability', predict_result['pos_probability'].cast(FloatType()))
prediction_result_topd = predict_result.select('mobile', 'label', 'predict', 'pos_probability').toPandas()

fpr, tpr, threshold = roc_curve(prediction_result_topd['label'], prediction_result_topd['pos_probability'],
                                pos_label=1)
ks_ary = tpr - fpr
ks = np.max(ks_ary)
print('ks_value:', ks)
max_ks_threshold = np.argmax(ks_ary)
print('threshold:', threshold[max_ks_threshold])
prediction_result_topd = prediction_result_topd.sort_values(by=['pos_probability'], axis=0, ascending=False)
prediction_result_topd['ks_predict'] = prediction_result_topd['pos_probability'].apply(
    lambda x: 1 if x >= threshold[max_ks_threshold] else 0)
y_true = prediction_result_topd['label']
y_pred = prediction_result_topd['predict']
y_scores = prediction_result_topd['pos_probability']
acc = accuracy_score(y_true=y_true, y_pred=y_pred)
# ban_acc = balanced_accuracy_score(y_true, y_pred, adjusted=True)
avg_precision = average_precision_score(y_true, y_scores)
pre = precision_score(y_true, y_pred, pos_label=1)
re = recall_score(y_true, y_pred, pos_label=1)
auc = roc_auc_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, pos_label=1)
fbeta = fbeta_score(y_true, y_pred, beta=0.5)
print(acc, avg_precision, pre, re, auc, f1, fbeta)

spark.stop()

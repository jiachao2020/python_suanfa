import os
import sys

from pyspark import SparkConf, SparkContext
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import split
from pyspark.sql.types import *

sc = SparkContext(appName='TF-IDF').getOrCreate()
sqlContext = SQLContext(sc)

final_2018 = sc.textFile("/user/hive/warehouse/rac.db/liantong_mobile_sentence_20181207/*")

final_2018 = final_2018.map(lambda row: row.strip().split('\001'))
srcdf = final_2018.toDF(["mobile", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=928)

idf = IDF(inputCol="rawFeatures", outputCol="features")
wordsData = tokenizer.transform(srcdf)

featurizedData = hashingTF.transform(wordsData)
idfModel = idf.fit(featurizedData)

rescaledData = idfModel.transform(featurizedData)
rescaledData = rescaledData.withColumn("TF-rawFeatures", rescaledData["rawFeatures"].cast(StringType()))
rescaledData = rescaledData.withColumn("TF-IDF-Features", rescaledData["features"].cast(StringType()))
rescaledData = rescaledData.withColumn("TF-words", rescaledData["words"].cast(StringType()))
result = rescaledData.select("features", "mobile").rdd.map(
    lambda x: Row(mobile=x['mobile'], features=Vectors.dense(x['features']))).toDF()

result = result.withColumn("features", result["features"].cast(StringType()))

result.write.csv(os.path.join("/user/chenzhixiang/data/credit/20181207_with_freq", 'features'), header=False)

result = sc.textFile(os.path.join("/user/chenzhixiang/data/credit/20181207_with_freq", 'features/p*'))
# result = result.map(lambda x:x.strip('"').strip('[').strip(']').split(',')).toDF([str(i) for i in range(600)])
result = result.map(
    lambda x: x.strip().split('",')[0].strip('"').strip('[').strip(']').split(',') + [x.strip().split('",')[1]]).toDF(
    [str(i) for i in range(928)] + ['mobile'])

result.repartition(1).write.csv(os.path.join("/user/chenzhixiang/data/credit/20181207_with_freq", 'features1'),
                                header=True)

sc.stop()

import os
import sys

from pyspark import SparkConf, SparkContext
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import Row, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import split
from pyspark.sql.types import StringType

'''spark = SparkSession\
        .builder\
        .appName("PythonALS")\
        .getOrCreate()
sc = spark.sparkContext'''
sc = SparkContext(appName="FPGrowth").getOrCreate()
sqlContext = SQLContext(sc)

# transactions = data.map(lambda line: line.strip().split(' '))
# model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=10)
# result = model.freqItemsets().collect()
# for fi in result:
# print(fi)

data = sqlContext.read.text("/user/hive/warehouse/rac.db/liantong_mobile_sentence_20190118/0*").select(
    split("value", ",").alias("items"))
# data.show(truncate=False)
fp = FPGrowth(minSupport=0.0008, minConfidence=1, itemsCol='items', predictionCol='prediction', numPartitions=1000)
fpm = fp.fit(data)
# freqitem = fpm.freqItemsets.withColumn("items", fpm.freqItemsets["items"].cast(StringType()))
freqitem = fpm.freqItemsets.rdd.map(lambda x: Row(items=sorted(x['items']), freq=x['freq'])).toDF().select('items',
                                                                                                           'freq')
freqitem = freqitem.withColumn("items", freqitem["items"].cast(StringType()))
freqitem.write.saveAsTable('rac.liantong_huanbei_all_20190121')
# freqitem.write.csv(
#     os.path.join("/user/chenzhixiang/dianxin_FP_Growth_20181217_weichika/", 'freItemsets'), header=True)

# fpm_asso = fpm.associationRules.withColumn("antecedent", fpm.associationRules["antecedent"].cast(StringType()))
# fpm_asso = fpm_asso.withColumn("consequent", fpm_asso["consequent"].cast(StringType()))
# fpm_asso.repartition(1).write.csv(os.path.join("/user/chenzhixiang/FP_Growth_20181207_weichika/", 'associationRules'),
#                                   header=True)
# new_data = sqlContext.createDataFrame([(["https://ecentre.spdbccc.com.cn"],)], ["items"])
# predict_data = fpm.transform(new_data)
# print(sorted(predict_data.first().prediction))
# predict_data = predict_data.withColumn("items", predict_data["items"].cast(StringType()))
# predict_data.withColumn("prediction", predict_data["prediction"].cast(StringType())).repartition(1).write.csv(
#     os.path.join("/user/chenzhixiang/FP_Growth_20181029/", 'predict_data'),
#     header=True)
sc.stop()

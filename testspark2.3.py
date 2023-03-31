import findspark
findspark.init()


from pyspark.sql import SparkSession
from pyspark import SparkConf
#from pyspark.ml.functions import array_to_vector

spark=SparkSession.builder.master("local[*]").appName("test").enableHiveSupport().getOrCreate()

sc=spark.sparkContext

#df1 = spark.createDataFrame([([1.5, 2.5],),], schema='v1 array<double>')
#df1.select(array_to_vector('v1').alias('vec1')).collect()

#print(df1)
data = ["hello", "world", "hello", "world"]

rdd = sc.parallelize(data,8)
res_rdd = rdd.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

print(res_rdd.first())
sc.stop()
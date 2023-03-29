# import findspark
# findspark.init('/path/to/spark/home')
#from pyspark import SparkConf,SparkContext
from pyspark.sql import SparkSession
#conf = SparkConf().setAppName('test').setMaster("local[*]")
spark=SparkSession.builder.master("local[*]").appName("test").getOrCreate()

data = ["hello", "world", "hello", "world"]

rdd = spark.sparkContext.parallelize(data)
res_rdd = rdd.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

print(res_rdd.first())
spark.stop()
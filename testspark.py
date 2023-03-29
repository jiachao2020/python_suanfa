# import findspark
# findspark.init('/path/to/spark/home')
from pyspark import SparkConf,SparkContext
conf = SparkConf().setAppName('test')
try:
    sc.stop()
except:
    pass
sc = SparkContext(conf = conf)

data = ["hello", "world", "hello", "world"]

rdd = sc.parallelize(data)
res_rdd = rdd.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

print(res_rdd.first())
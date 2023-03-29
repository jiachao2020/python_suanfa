from pyspark.sql import SparkSession
from pyspark import SparkConf

conf=SparkConf()
#spark.conf.set("spark.executor.memory","5g")
config=(("spark.app.name","test"),
        ("spark.executor.memory","6g"),
        ("spark.master","local[*]")
)
#spark=SparkSession.builder.master("local[*]").appName("test").enableHiveSupport().getOrCreate()
conf.setAll(config)
spark=SparkSession.builder.config(conf=conf).getOrCreate()
sc=spark.sparkContext



data = ["hello", "world", "hello", "world"]

rdd = sc.parallelize(data,8)
res_rdd = rdd.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

print(res_rdd.first())
sc.stop()
import findspark
findspark.init()

from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import col
sc =SparkContext()
sqlContext = SQLContext(sc)
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true',
inferschema='true').load('/Users/jiachao/Downloads/sf-crime/train.csv')
drop_list = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
data = data.select([column for column in data.columns if column not in drop_list])
#data.show(5)
#data.printSchema()
# data.groupBy("Category") \
#     .count() \
#     .orderBy(col("count").desc()) \
#     .show()
data.groupBy("Category") \
.count() \
    .orderBy(col("count").desc()) \
    .show()
import findspark

findspark.init()

from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import col

sc = SparkContext()
sqlContext = SQLContext(sc)
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true',
                                                                  inferschema='true').load(
    '/Users/jiachao/Downloads/sf-crime/train.csv')
drop_list = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
data = data.select([column for column in data.columns if column not in drop_list])
# data.show(5)
# data.printSchema()
# data.groupBy("Category") \
#     .count() \
#     .orderBy(col("count").desc()) \
#     .show()
# data.groupBy("Category") \
# .count() \
#     .orderBy(col("count").desc()) \
#     .show()


from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="words", pattern="\W")
# stop words
add_stopwords = ["http", "https", "amp", "rt", "t", "c", "the"]
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

'''StringIndexer

StringIndexer将一列字符串label编码为一列索引号（从0到label种类数-1），根据label出现的频率排序，最频繁出现的label的index为0。

在该例子中，label会被编码成从0到32的整数，最频繁的 label(LARCENY/THEFT) 会被编码成0。
'''
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

label_stringIdx = StringIndexer(inputCol="Category", outputCol="label")
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors,
                            label_stringIdx])
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)

# 训练/测试数据集划分
# set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript", "Category", "probability", "label", "prediction") \
    .orderBy("probability", ascending=False) \
    .show(n=10, truncate=30)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(evaluator.evaluate(predictions))

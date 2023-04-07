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





#使用交叉验证来优化参数，基于词频特征的逻辑回归模型进行优化
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)

# 训练/测试数据集划分
# set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 90)
#随机数种子 100 263383
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))
#网格搜索，交叉验证
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder() \
             .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter \
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) \
                  # Elastic Net Parameter (Ridge = 0)
#            .addGrid(model.maxIter, [10, 20, 50]) #Number of iterations
#            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())
# Create 5-fold CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
cvModel = cv.fit(trainingData)
predictions = cvModel.transform(testData)

# Evaluate best model
print(evaluator.evaluate(predictions))

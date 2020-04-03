
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import desc, size, max, abs
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import pyspark.ml.evaluation as ev
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.linalg import Vectors
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np

def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
    #auc = roc_auc_score(y_test, y_scores)
    return [accuracy, precision, recall, f1]

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

seed=1

spark = init_spark()
# Load training data
#filepath='Data/2018_Financial_Data_spark.csv'
filepath='Data/2018_Financial_Data.csv'
#training = spark.read.format("libsvm").load('Data/2018_Financial_Data_spark.csv')
data=spark.read.csv(filepath,header='true',inferSchema='true',sep=',')

#data=data.rdd.map(lambda x:(Vectors.dense(x[1:-1]), x[0])).toDF(["features", "label"])
data=data.rdd.map(lambda x:(Vectors.dense(x[1:-4]), x[-1])).toDF(["features", "label"])

(trainingData, testData) = data.randomSplit([0.7, 0.3],seed=seed)


lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8,tol=0.0001, family="binomial")
dt = DecisionTreeClassifier(seed=seed)
# Fit the model
#mlrModel = lr.fit(training)

pipeline = Pipeline(stages=[lr])
pipelineDt = Pipeline(stages=[dt])

model = pipelineDt.fit(trainingData)
test_model = model.transform(testData)

label=testData.select("label").toPandas().values
predict=test_model.select("prediction").toPandas().values

print("[accuracy,precision,recall,f1]")
print(evaluate(label,predict))
np.savetxt('res/predictedDT.txt', predict, fmt='%01d')

import pyspark.ml.tuning as tune

md=np.arange(1,50,10)
grid = tune.ParamGridBuilder()\
    .addGrid(dt.maxDepth, [1, 10, 20, 30])\
    .build()

evaluator = ev.BinaryClassificationEvaluator(
    rawPredictionCol='probability',
    labelCol='label'
)

# 使用K-Fold交叉验证评估各种参数的模型
cv = tune.CrossValidator(
    estimator=dt,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    numFolds=3
)


pipelineDtCV = Pipeline(stages=[cv])
cvModel = cv.fit(trainingData)
results = cvModel.transform(testData)

label=results.select("label").toPandas().values
predict=results.select("prediction").toPandas().values

print("[accuracy,precision,recall,f1]")
print(evaluate(label,predict))


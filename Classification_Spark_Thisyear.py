
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import NaiveBayes

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
import pyspark.ml.tuning as tune

#predefined functions
def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
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

# Load data
#filepath='Data/2018_Financial_Data_spark.csv'
filepath='Data/2018_Financial_Data.csv'

data=spark.read.csv(filepath,header='true',inferSchema='true',sep=',')
data=data.rdd.map(lambda x:(Vectors.dense(x[1:-4]), x[-1])).toDF(["features", "label"])

(trainingData, testData) = data.randomSplit([0.8, 0.2],seed=seed)

#model init
lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8,tol=0.0001, family="binomial")
dt = DecisionTreeClassifier(seed=seed)
rf=RandomForestClassifier(seed=seed)
GBDT=GBTClassifier(seed=seed)
MLP = MultilayerPerceptronClassifier(maxIter=100,blockSize=128, seed=seed)
SVM=LinearSVC(regParam=0.1)
nb=NaiveBayes(smoothing=1.0, modelType="binomial")

#model training and testing functions
def LR(trainingData,testData):

    Model = lr.fit(trainingData)
    results = Model.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedLR_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")

    return evaluate(label,predict)

def SVM(trainingData,testData):

    Model = SVM.fit(trainingData)
    results = Model.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedSVM_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")

    return evaluate(label,predict)

def NB(trainingData,testData):

    Model = nb.fit(trainingData)
    results = Model.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedLR_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")

    return evaluate(label,predict)

def DTclf(trainingData,testData):

    grid = tune.ParamGridBuilder()\
        .addGrid(dt.maxDepth, [1, 10, 20, 30])\
        .build()

    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol='probability',
        labelCol='label'
    )

    # 3-fold validation
    cv = tune.CrossValidator(
        estimator=dt,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3
    )


    #pipelineDtCV = Pipeline(stages=[cv])
    cvModel = cv.fit(trainingData)
    results = cvModel.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedDT_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")
    #print(evaluate(label,predict))
    return evaluate(label,predict)

def RFclf(trainingData,testData):

    grid = tune.ParamGridBuilder()\
        .addGrid(rf.maxDepth, [1, 10, 20, 30]) \
        .addGrid(rf.numTrees, [1, 10, 20, 30,40,50]) \
        .build()

    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol='probability',
        labelCol='label'
    )

    # 3-fold validation
    cv = tune.CrossValidator(
        estimator=rf,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3
    )


    #pipelineDtCV = Pipeline(stages=[cv])
    cvModel = cv.fit(trainingData)
    results = cvModel.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedRF_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")
    #print(evaluate(label,predict))
    return evaluate(label,predict)

def GBDTclf(trainingData, testData):


    max_depth=[1,5,10]
    grid = tune.ParamGridBuilder() \
        .addGrid(GBDT.maxDepth, max_depth) \
        .build()

    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol='probability',
        labelCol='label'
    )

    # 3-fold validation
    cv = tune.CrossValidator(
        estimator=GBDT,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3
    )

    # pipelineDtCV = Pipeline(stages=[cv])
    cvModel = cv.fit(trainingData)
    results = cvModel.transform(testData)

    label = results.select("label").toPandas().values
    predict = results.select("prediction").toPandas().values
    np.savetxt('res/predictedGBDT_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")
    # print(evaluate(label,predict))
    return evaluate(label, predict)

def MLPclf(trainingData, testData):
    layers = [[10,5],[10,5,3]]

    grid = tune.ParamGridBuilder() \
        .addGrid(MLP.layers, layers) \
        .build()

    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol='probability',
        labelCol='label'
    )

    # 3-fold validation
    cv = tune.CrossValidator(
        estimator=MLP,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3
    )

    # pipelineDtCV = Pipeline(stages=[cv])
    cvModel = cv.fit(trainingData)
    results = cvModel.transform(testData)

    label = results.select("label").toPandas().values
    predict = results.select("prediction").toPandas().values
    np.savetxt('res/predictedMLP_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")
    # print(evaluate(label,predict))
    return evaluate(label, predict)


print('LogisticRegression:')
print(LR(trainingData,testData))
print('NaiveBayes:')
print(NB(trainingData,testData))
print('SVM:')
print(SVM(trainingData,testData))
print('DecisionTreeClassifier:')
print(DTclf(trainingData,testData))
print('RandomForestClassifier')
print(RFclf(trainingData,testData))
print('GBTClassifier')
print(GBDTclf(trainingData,testData))
print('MultilayerPerceptronClassifier:')
print(MLPclf(trainingData,testData))
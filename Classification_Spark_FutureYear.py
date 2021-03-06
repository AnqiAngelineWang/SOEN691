
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.mllib.classification import SVMModel
from pyspark.mllib.classification import SVMWithSGD
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
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import MinMaxScaler as Scaler
import pdb


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
filepath1='Data/2014_Financial_Data.csv'
filepath2='Data/2015_Financial_Data.csv'

train=spark.read.csv(filepath1,header='true',inferSchema='true',sep=',')
test=spark.read.csv(filepath2,header='true',inferSchema='true',sep=',')
#data=spark.read.csv(filepath,header='false',inferSchema='true',sep=',')
#find null

feature_number=len(train.columns)-5

trainingData=train.fillna(0)
trainingData=trainingData.rdd.map(lambda x:(Vectors.dense(x[1:-4]), x[-1])).toDF(["features", "label"])

testData=test.fillna(0)
testData=testData.rdd.map(lambda x:(Vectors.dense(x[1:-4]), x[-1])).toDF(["features", "label"])


#(trainingData, testData) = data.randomSplit([0.8, 0.2],seed=seed)


scaler = Scaler(inputCol="features", outputCol="scaledFeatures")
scalerModel = scaler.fit(trainingData)
trainingData = scalerModel.transform(trainingData)
testData = scalerModel.transform(testData)


#pdb.set_trace()
#model init
lr = LogisticRegression(featuresCol='scaledFeatures',maxIter=100, regParam=0.3, elasticNetParam=0.8,tol=0.0001, family="binomial")
dt = DecisionTreeClassifier(featuresCol='scaledFeatures',seed=seed)
rf=RandomForestClassifier(featuresCol='scaledFeatures',seed=seed)
GBDT=GBTClassifier(featuresCol='scaledFeatures',seed=seed)
layers = [feature_number,10,2]
mlp = MultilayerPerceptronClassifier(featuresCol='scaledFeatures',layers=layers, seed=seed)
svm=LinearSVC(featuresCol='scaledFeatures',regParam=0.1)
nb=NaiveBayes(featuresCol='scaledFeatures',smoothing=1.0)


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

    Model = svm.fit(trainingData)
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



    mlp=MultilayerPerceptronClassifier().setFeaturesCol("features").setLabelCol("label").setLayers(layers).setSolver("gd").setStepSize(0.3) .setMaxIter(1000)

    mlpModel = mlp.fit(trainingData)
    results = mlpModel.transform(testData)

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
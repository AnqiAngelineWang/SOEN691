
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
from pyspark.sql.functions import col, explode, array, lit
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
import time

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
#data=spark.read.csv(filepath,header='false',inferSchema='true',sep=',')
#find null
#data.rdd.map(lambda row:(row['id'],sum([c==None for c in row]))).collect()
feature_number=len(data.columns)-5
data=data.fillna(0)
cols=data.columns[1:-4]
data=data.rdd.map(lambda x:(Vectors.dense(x[1:-4]), x[-1])).toDF(["features", "label"])
#data=data.rdd.map(lambda x:(Vectors.dense(x[1:-1]), x[0])).toDF(["features", "label"])



(trainingData, testData) = data.randomSplit([0.8, 0.2],seed=seed)


scaler = Scaler(inputCol="features", outputCol="scaledFeatures")
scalerModel = scaler.fit(trainingData)
trainingData = scalerModel.transform(trainingData)
testData = scalerModel.transform(testData)


#pdb.set_trace()
#model init
lr = LogisticRegression(featuresCol='scaledFeatures',maxIter=100, regParam=0.3, elasticNetParam=0.8,tol=0.0001, family="binomial")
dt = DecisionTreeClassifier(featuresCol='scaledFeatures',seed=seed)
rf=RandomForestClassifier(featuresCol='scaledFeatures',seed=seed,numTrees=20)
GBDT=GBTClassifier(featuresCol='scaledFeatures',seed=seed)
layers = [feature_number,10,5,2]
mlp = MultilayerPerceptronClassifier(featuresCol='scaledFeatures',layers=layers, seed=seed)
svm=LinearSVC(featuresCol='scaledFeatures',regParam=0.1)
nb=NaiveBayes(featuresCol='scaledFeatures',smoothing=1.0)

times=[]
#model training and testing functions
def LR(trainingData,testData):

    start = time.time()
    Model = lr.fit(trainingData)
    end = time.time()
    times.append(end - start)

    results = Model.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedLR_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")

    return evaluate(label,predict)

def SVM(trainingData,testData):
    start = time.time()
    Model = svm.fit(trainingData)
    end = time.time()
    times.append(end - start)
    results = Model.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedSVM_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")

    return evaluate(label,predict)

def NB(trainingData,testData):
    start = time.time()
    Model = nb.fit(trainingData)
    end = time.time()
    times.append(end - start)
    results = Model.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedLR_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")

    return evaluate(label,predict)

def DTclf(trainingData,testData):




    #pipelineDtCV = Pipeline(stages=[cv])
    start = time.time()
    cvModel = dt.fit(trainingData)
    end = time.time()
    times.append(end - start)

    results = cvModel.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedDT_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")
    #print(evaluate(label,predict))
    return evaluate(label,predict)

def RFclf(trainingData,testData):


    start = time.time()
    cvModel = rf.fit(trainingData)
    end = time.time()
    times.append(end - start)
    results = cvModel.transform(testData)

    label=results.select("label").toPandas().values
    predict=results.select("prediction").toPandas().values
    np.savetxt('res/predictedRF_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")
    #print(evaluate(label,predict))
    return evaluate(label,predict)

def GBDTclf(trainingData, testData):



    start = time.time()
    cvModel = GBDT.fit(trainingData)
    end = time.time()
    times.append(end - start)
    results = cvModel.transform(testData)

    label = results.select("label").toPandas().values
    predict = results.select("prediction").toPandas().values
    np.savetxt('res/predictedGBDT_spark.txt', predict, fmt='%01d')
    print("[accuracy,precision,recall,f1]")
    # print(evaluate(label,predict))
    return evaluate(label, predict)

def MLPclf(trainingData, testData):



    mlp=MultilayerPerceptronClassifier().setFeaturesCol("features").setLabelCol("label").setLayers(layers).setSolver("gd").setStepSize(0.3) .setMaxIter(1000)
    start = time.time()
    mlpModel = mlp.fit(trainingData)
    end = time.time()
    times.append(end - start)
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

print("Running Time:")
print("LR,NB,SVM,DT,RF,GBDT,MLP")
print(times)
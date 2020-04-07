import numpy as np
import sklearn
import pandas as pd
import os

from numpy import random
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble
from sklearn import svm

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
#import xgboost as xgb
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error,mean_absolute_error, roc_curve, classification_report, auc)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import time


from numpy.random import seed
seed(1)

seed=1

filename='Data/2018_Financial_Data.csv'
print(filename)

data=pd.read_csv(filename, header=0)

data = pd.DataFrame(data)

print(data.iloc[:, -1].unique().size)

if data.iloc[:, -1].unique().size==2:
    def evaluate(y_test, y_pred, y_scores):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
        auc = roc_auc_score(y_test, y_scores)
        return [accuracy, precision, recall, f1, auc]
else:

    def evaluate(y_test, y_pred, y_scores):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', pos_label=1)
        recall = recall_score(y_test, y_pred, average='weighted', pos_label=1)
        f1 = f1_score(y_test, y_pred, average='weighted', pos_label=1)
        #    auc = roc_auc_score(y_test, y_scores)
        return [accuracy, precision, recall, f1, auc]



data.fillna(0, inplace=True)
'''
# Categorical boolean mask
categorical_feature_mask = data.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = data.columns[categorical_feature_mask].tolist()
print(categorical_cols)
#Use LabelEncoder() to transfer categorical data to numurical data
le=LabelEncoder()
print(len(categorical_cols))
if len(categorical_cols)!=0:
    data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
#data[categorical_cols].head(10)
'''



data=np.array(data)


#first column is ID
X,y=data[:,1:-4], data[:, -1]


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train2 = scaler.transform(X_train)
X_test2 = scaler.transform(X_test)



traindata = np.array(X_train2)
trainlabel = np.array(y_train)

testdata = np.array(X_test2)
testlabel = np.array(y_test)

traindata=traindata.astype(float)
trainlabel=trainlabel.astype(int)
testdata=testdata.astype(float)
testlabel = testlabel.astype(int)

times=[]

def MLP(X_train,y_train,X_test,y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=seed)

    start = time.time()
    clf = clf.fit(X_train,y_train)
    end = time.time()
    times.append(end - start)

    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]

    #print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedMLP.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)

def LogisticRegression(X_train,y_train,X_test,y_test):
    clf = sklearn.linear_model.LogisticRegression(random_state=seed, solver='lbfgs')
    start = time.time()
    clf = clf.fit(X_train, y_train)
    end = time.time()
    times.append(end - start)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedLR.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)



def CART(X_train,y_train,X_test,y_test):
    #clf = tree.DecisionTreeClassifier(max_depth=5)
    #clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=5)

    clf = tree.DecisionTreeClassifier(max_depth=5,random_state=seed)

    start = time.time()
    clf = clf.fit(X_train, y_train)
    end = time.time()
    times.append(end - start)

    #clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedDTbase.txt', y_pred, fmt='%01d')
    #print("best parameters:", gridcv.best_params_)
#    tree.plot_tree(clf.fit(X_train,y_train))
    #tree.export_graphviz()
    #dot_data = tree.export_graphviz(clf, out_file=None)
    #graph = graphviz.Source(dot_data)
    #graph.render("tree")
    return evaluate(y_test,y_pred,y_scores)

#randomforest

def RandomForest(X_train,y_train,X_test,y_test):
    clf = ensemble.RandomForestClassifier(max_features=None,max_depth=5,n_estimators=20,random_state=seed)

    #clf = sklearn.model_selection.GridSearchCV(clf, param_grid, verbose=1, cv=3)
    start = time.time()
    clf = clf.fit(X_train, y_train)
    end = time.time()
    times.append(end - start)

    #gridcv = gridcv.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    #print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedRFbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)



def SVM(X_train,y_train,X_test,y_test):

    clf = sklearn.svm.SVC(kernel='rbf')

    start = time.time()
    clf = clf.fit(X_train, y_train)
    end = time.time()
    times.append(end - start)

    y_pred = clf.predict(X_test)
    y_scores = clf.decision_function(X_test)
    #print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedSVMNormbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)


def Gaussian_NB(X_train,y_train,X_test,y_test):
    clf = GaussianNB()

    start = time.time()
    clf = clf.fit(X_train, y_train)
    end = time.time()
    times.append(end - start)

    y_pred = clf.predict(X_test)
    y_scores  = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]

    np.savetxt('res/predictedNBbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)

def kNN(X_train,y_train,X_test,y_test):
    clf = KNeighborsClassifier()

    start = time.time()
    clf = clf.fit(X_train, y_train)
    end = time.time()
    times.append(end - start)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    #print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedKNNbase.txt', y_pred, fmt='%01d')

    return evaluate(y_test,y_pred,y_scores)


def GBDT(X_train,y_train,X_test,y_test):

    clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=seed)
    #clf=clf.fit(X_train, y_train)

    start = time.time()
    clf = clf.fit(X_train, y_train)
    end = time.time()
    times.append(end - start)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    #print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedBGDTbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)


#SVM-rbf
#knn odd num
print("LR training")
a5=LogisticRegression(traindata,trainlabel,testdata,testlabel)
print(a5)
print("NB training")
a2=Gaussian_NB(traindata,trainlabel,testdata,testlabel)
print(a2)
#a3=xgboost(traindata,trainlabel,testdata,testlabel)
print("SVM training")
a6=SVM(traindata,trainlabel,testdata,testlabel)
print(a6)
print("DT training")
a10=CART(traindata,trainlabel,testdata,testlabel)
print(a10)
print("RF training")
a8=RandomForest(traindata,trainlabel,testdata,testlabel)
print(a8)
print("GBDT training")
a1=GBDT(traindata,trainlabel,testdata,testlabel)
print(a1)
print("MLP training")
a7=MLP(traindata,trainlabel,testdata,testlabel)
print(a7)
res = [a1, a2, a5, a6, a7, a8, a10]



print("Running time:")
print("LR,NB,SVM,DT,RF,GBDT,MLP")
print(times)



def tsneShow(X,y,Xt,yt):
    yys = np.array(y)
    yyt = np.array(yt)
    for i in range(0, yyt.size):
        if yyt[i] == 0:
            yyt[i] = 2
        elif yyt[i] == 1:
            yyt[i] = 3

    H = np.vstack((X, Xt))
    Y = np.concatenate((yys, yyt), axis=0)

    #pca = PCA().fit_transform(H)
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(H)

    markers = ('.')
    plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=Y, marker=markers[0], cmap=plt.cm.gist_rainbow)

    plt.colorbar()
    plt.show()
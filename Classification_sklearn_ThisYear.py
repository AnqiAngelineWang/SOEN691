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

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from numpy.random import seed
seed(1)

seed=1

filename='Data/2014_Financial_Data.csv'
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

def MLP(X_train,y_train,X_test,y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=seed)
    l=X_train[0].size
    lr = np.logspace(-5,-1, 5)
    #print(lr)
    #print(X_train[0].size)
    hz=[(10, 5),(10,5,3)]
    param_grid = {'alpha': lr,'hidden_layer_sizes':hz}

    gridcv = sklearn.model_selection.GridSearchCV(clf, param_grid, verbose=1, cv=3)
    gridcv.fit(X_train, y_train)

    gridcv = gridcv.fit(X_train,y_train)
    y_pred = gridcv.best_estimator_.predict(X_test)
    y_scores = gridcv.best_estimator_.predict_proba(X_test)[:,1]

    print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedMLP.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)

def LogisticRegression(X_train,y_train,X_test,y_test):
    clf = sklearn.linear_model.LogisticRegression(random_state=seed, solver='lbfgs')
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedLR.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)



def CART(X_train,y_train,X_test,y_test):
    #clf = tree.DecisionTreeClassifier(max_depth=5)
    #clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=5)

    clf = tree.DecisionTreeClassifier(random_state=seed)

    md=np.arange(1,50,10)
    param_grid = {'max_depth': md}

    gridcv = sklearn.model_selection.GridSearchCV(clf, param_grid, verbose=1, cv=3)
    gridcv.fit(X_train, y_train)

    #clf = clf.fit(X_train,y_train)
    y_pred = gridcv.best_estimator_.predict(X_test)
    y_scores = gridcv.best_estimator_.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedDTbase.txt', y_pred, fmt='%01d')
    print("best parameters:", gridcv.best_params_)
#    tree.plot_tree(clf.fit(X_train,y_train))
    #tree.export_graphviz()
    #dot_data = tree.export_graphviz(clf, out_file=None)
    #graph = graphviz.Source(dot_data)
    #graph.render("tree")
    return evaluate(y_test,y_pred,y_scores)

#randomforest

def RandomForest(X_train,y_train,X_test,y_test):
    clf = ensemble.RandomForestClassifier(max_features=None,random_state=seed)

    md = np.arange(1, 50, 10)
    ne = np.arange(1, 50, 10)
    param_grid = {'max_depth': md,'n_estimators':ne}

    gridcv = sklearn.model_selection.GridSearchCV(clf, param_grid, verbose=1, cv=3)
    gridcv.fit(X_train, y_train)

    gridcv = gridcv.fit(X_train,y_train)
    y_pred = gridcv.best_estimator_.predict(X_test)
    y_scores = gridcv.best_estimator_.predict_proba(X_test)[:,1]
    print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedRFbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)



def svm_one_class(X_train,y_train,X_test,y_test):
    _train = preprocessing.normalize(X_train, norm='l2')
    _test = preprocessing.normalize(X_test, norm='l2')
    lin_clf = svm.LinearSVC()
    lin_clf.fit(_train, y_train)
    LinearSVC(C=1000, class_weight=None, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=seed, tol=0.0001,
     verbose=0)
    y_pred = lin_clf.predict(_test)
    y_scores = lin_clf.decision_function(_test)
    np.savetxt('res/predictedSVMbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)


def SVM(X_train,y_train,X_test,y_test):

    svm = sklearn.svm.SVC(kernel='rbf')
    C_grid = np.logspace(0, 3, 4)
    gamma_grid = np.logspace(-2, 1, 4)
    param_grid = {'C': C_grid, 'gamma': gamma_grid}
    gridcv = sklearn.model_selection.GridSearchCV(svm, param_grid, verbose=1, cv=3)
    gridcv.fit(X_train, y_train)

    y_pred = gridcv.best_estimator_.predict(X_test)
    y_scores = gridcv.best_estimator_.decision_function(X_test)
    print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedSVMNormbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)


def Gaussian_NB(X_train,y_train,X_test,y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    y_scores  = gnb.fit(X_train, y_train).predict_proba(X_test)[:,1]

    np.savetxt('res/predictedNBbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)

def kNN(X_train,y_train,X_test,y_test):
    clf = KNeighborsClassifier()
    #n_neighbors=5
    nn=np.arange(1,100,10)
    param_grid = {'n_neighbors': nn}

    gridcv = sklearn.model_selection.GridSearchCV(clf, param_grid, verbose=1, cv=3)
    gridcv.fit(X_train, y_train)
    y_pred = gridcv.best_estimator_.predict(X_test)
    y_scores = gridcv.best_estimator_.predict_proba(X_test)[:,1]
    print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedKNNbase.txt', y_pred, fmt='%01d')

    return evaluate(y_test,y_pred,y_scores)

def AdaBoost(X_train,y_train,X_test,y_test):
   # dt_stump = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)
    dt_stump = DecisionTreeClassifier()
    dt_stump.fit(X_train, y_train)
    dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

    clf = ensemble.AdaBoostClassifier(base_estimator=dt_stump,learning_rate=0.1,algorithm='SAMME')
    clf=clf.fit(X_train, y_train)


    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedABbase.txt', y_pred, fmt='%01d')

    return evaluate(y_test,y_pred,y_scores)

def GBDT(X_train,y_train,X_test,y_test):
    n_estimators = [10,50,100]
    learning_rate = [0.01,0.1,1]
    max_depth=[1,5,10]
    param_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate,'max_depth':max_depth}

    clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=seed)
    #clf=clf.fit(X_train, y_train)

    gridcv = sklearn.model_selection.GridSearchCV(clf, param_grid, verbose=1, cv=3)
    gridcv.fit(X_train, y_train)
    y_pred = gridcv.best_estimator_.predict(X_test)
    y_scores = gridcv.best_estimator_.predict_proba(X_test)[:,1]
    print("best parameters:", gridcv.best_params_)
    np.savetxt('res/predictedBGDTbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)


#SVM-rbf
#knn odd num
print("GBDT training")
a1=GBDT(traindata,trainlabel,testdata,testlabel)
print(a1)
print("NB training")
a2=Gaussian_NB(traindata,trainlabel,testdata,testlabel)
print(a2)
#a3=xgboost(traindata,trainlabel,testdata,testlabel)
print("kNN training")
a4=kNN(traindata,trainlabel,testdata,testlabel)
print(a4)
print("LR training")
a5=LogisticRegression(traindata,trainlabel,testdata,testlabel)
print(a5)
print("SVM training")
a6=SVM(traindata,trainlabel,testdata,testlabel)
print(a6)
print("MLP training")
a7=MLP(traindata,trainlabel,testdata,testlabel)
print(a7)
print("RF training")
a8=RandomForest(traindata,trainlabel,testdata,testlabel)
print(a8)
print("DT training")
a10=CART(traindata,trainlabel,testdata,testlabel)
print(a10)

res = [a1, a2, a4, a5, a6, a7, a8, a10]

# str(res)
#print(res)
#np.savetxt('res/' + 'ClassificationData6.Steel Plates FaultsFaults' + '.txt', res, fmt='%.4f')

#a12=svm_one_class(traindata,trainlabel,testdata,testlabel)
print("GBDT")
print("[accuracy,precision,recall,f1,auc]")
print(a1)

print("Naive_bayes")
print("[accuracy,precision,recall,f1,auc]")
print(a2)

#print("xgboost")
#print("[accuracy,precision,recall,f1,auc]")
#print(a3)

print("Knn")
print("[accuracy,precision,recall,f1,auc]")
print(a4)

print("LogisticRegression")
print("[accuracy,precision,recall,f1,auc]")
print(a5)

print("SVM")
print("[accuracy,precision,recall,f1,auc]")
print(a6)


print("MLP")
print("[accuracy,precision,recall,f1,auc]")
print(a7)

print("RandomForest")
print("[accuracy,precision,recall,f1,auc]")
print(a8)

print("CART")
print("[accuracy,precision,recall,f1,auc]")
print(a10)


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
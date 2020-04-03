import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def CountNA(data):
    '''
    统计每个变量的缺失值占比
    '''
    cols = data.columns.tolist()    #cols为data的所有列名
    n_df = data.shape[0]    #n_df为数据的行数
    for col in cols:
        missing = np.count_nonzero(data[col].isnull().values)  #col列中存在的缺失值个数
        mis_perc = float(missing) / n_df * 100
        print("{col}的缺失比例是{miss}%".format(col=col,miss=mis_perc))
# CountNA(data)
#---------------------------------------------------delete null value-------------------------------------------
def delete_null_value(data):
    data.dropna(axis=0,how="any",inplace=True)  #axis=0代表'行','any'代表任何空值行,若是'all'则代表所有值都为空时，才删除该行
    # data.dropna(axis=0,inplace=True)  #删除带有空值的行
    data.dropna(axis=1,inplace=True)  #删除带有空值的列

#----------------------------------------------------fill null value--------------------------------------------
def fill_with_freq(data,missing_col):
    for i in missing_col:
        freq_port = data[i].dropna().mode()[0]  # mode返回出现最多的数据,col_name为列名
        data[i] = data[i].fillna(freq_port)   #采用出现最频繁的值插补
    return data

def fill_with_median(data,missing_col):
    for i in missing_col:
        data[i].fillna(data[i].dropna().median(), inplace=True)  # 中位数插补，适用于偏态分布或者有离群点的分布
    return data

def fill_with_mean(data, missing_col):
    for i in missing_col:
        data[i].fillna(data[i].dropna().mean(), inplace=True)  # 均值插补，适用于正态分布
    return data

#------------------------------------------------------outlier-------------------------------------------------

# def outlier(data):
#     '''
#     当数据服从正态分布时，99.7%的数值应该位于距离均值3个标准差之内的距离，P(|x−μ|>3σ)≤0.003
#     '''
#     for i in data.columns:
#         neg_list=data[i]
#         for item in neg_list:
#             data[item + '_zscore'] = (data[item] - data[item].mean()) / data[item].std()
#             z_abnormal = abs(data[item + '_zscore']) > 3
#             print(item + '中有' + str(z_abnormal.sum()) + '个异常值')

#-------------------------------------------------------normalization------------------------------------------
def minmax(X_train, X_test, y_train):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    # 特征归一化
    x_train_sca = x_scaler.fit_transform(X_train)
    x_test_sca = x_scaler.transform(X_test)
    y_train_sca = y_scaler.fit_transform(pd.DataFrame(y_train))
    return x_train_sca,x_test_sca,y_train_sca

def statndard(X_train, X_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train2 = scaler.transform(X_train)
    X_test2 = scaler.transform(X_test)
    return X_train2,X_test2


filename='../Data/2014_Financial_Data.csv'
data=pd.read_csv(filename, header=0)
data = pd.DataFrame(data)
# print(data.isnull().sum()) #统计每列有几个缺失值
missing_col = data.columns[data.isnull().any()].tolist() #找出存在缺失值的列
# print(len(missing_col))
data=fill_with_median(data,missing_col)
seed=1
data=np.array(data)
#first column is ID
X,y=data[:,1:-4], data[:, -1]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)
X_train1,X_test1=statndard(X_train,X_test)
print(X_train1,X_test1)
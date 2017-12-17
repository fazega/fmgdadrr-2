# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:17:27 2017

@author: Michel
"""

from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor

import time


imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)


#
#Importe les données CSV en remplaçant les -1 par np.nan
def initialise_dataset(lien, n):

    dataset = read_csv(lien, header=None)

    header=[dataset[j][0] for j in range(n)]
    dataset_sansheader=[dataset[j][1:].astype(float) for j in range(n)]
    dataset_sansheader=np.array(dataset_sansheader)
    dataset_sansheader = pd.DataFrame(dataset_sansheader)
    dataset_sansheader=dataset_sansheader.replace(-1.0000,np.nan)
    return dataset_sansheader,header


def analyse(test, train, cv_prop, type="MLP"):
    t = time.time()
    cv_train = int(cv_prop*len(train))
    print(cv_train)
    entries = train.iloc[0:cv_train, 2:]
    results = [float(x) for x in list(train.iloc[0:cv_train]["is_churn"])]
    clf = None
    if(type == "MLP"):
        clf = MLPRegressor(solver='adam', alpha=1e-8, activation='logistic', tol=1e-8, hidden_layer_sizes=(30,10,20))
    elif(type == "KN"):
        clf = KNeighborsRegressor(3, weights='distance')
    elif(type == "TREE"):
        clf = tree.DecisionTreeRegressor() #DON'T WORK, CLASSIFIER ?
    elif(type == "LINEAR"):
        clf = BayesianRidge()
    elif(type == "RBF"):
        clf = SVR(kernel='rbf', C=1e3, gamma='auto')
    elif(type == "XGBOOST"):
        clf = XGBRegressor(objective='binary:logistic',
            n_estimators=300,
            min_child_weight=10.0,
            max_depth=7,
            max_delta_step=1.8,
            colsample_bytree=0.4,
            subsample=0.8,
            learning_rate=0.025,
            gamma=0.65)
        #xgb_params = {'eta': 0.03,
        #      'max_depth': 7,
        #      'subsample': 1.0,
        #      'colsample_bytree': 0.4,
        #      'min_child_weight': 10,
        #      'objective': 'binary:logistic',
        #      'eval_metric': 'auc',
        #      'seed': 99,
        #      'silent': True}
        #d_train = xgb.DMatrix(entries, results)
        #d_valid = xgb.DMatrix(train[cv_train:,2:],train[cv_train:,1])

        #watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        #clf = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=100, early_stopping_rounds=200)
    M = entries.as_matrix()
    for i in range(len(M)):
        for j in range(len(M[0])):
            if(M[i,j] == "nan"):
                M[i,j] = 0.
            else:
                M[i,j]=float(M[i,j])
    clf.fit(M, results)

    #cv_result = clf.predict(train.iloc[cv_train:, 2:])

    toTest = test.iloc[0:,2:]
    M = toTest.as_matrix()
    for i in range(len(M)):
        for j in range(len(M[0])):
            if(M[i,j] == "nan"):
                M[i,j] = 0.
            else:
                M[i,j]=float(M[i,j])
    Z = clf.predict(M)
    print(str(time.time()-t)+" seconds to achieve "+type+".")

    r = pd.DataFrame(columns=["msno", "is_churn"])
    for i in range(len(Z)):
        r.loc[i] = [test.index.values[i],max(0,Z[i])]
    return r


def export_csv(result):
    result.to_csv('result.csv', index=False)

def gather_dataset():
    users = {}
    headers = ["is_churn"]

    train_csv = read_csv("trainT.csv", header=None)
    for i in range(len(train_csv[0])):
        users[train_csv[0][i]] = {"is_churn":train_csv[1][i]}

    members_csv = read_csv("membersT.csv", header=None)
    headers += list(members_csv.loc[0][1:])
    for i in range(1,len(members_csv.columns)):
        for j in range(1,len(members_csv[i])):
            if(not members_csv[0][j] in users):
                users[members_csv[0][j]] = {}
            users[members_csv[0][j]][members_csv[i][0]] = members_csv[i][j]

    transactions_csv = read_csv("transactionsT.csv", header=None)
    headers += list(transactions_csv.loc[0][1:])
    for i in range(1,len(transactions_csv.columns)):
        for j in range(1,len(transactions_csv[i])):
            if(not transactions_csv[0][j] in users):
                users[transactions_csv[0][j]] = {"is_churn":"NaN"}
            users[transactions_csv[0][j]][transactions_csv[i][0]] = transactions_csv[i][j]
    print(headers)
    dataset = pd.DataFrame(columns=headers, index=users.keys())
    for u in users:
        for h in headers:
            if(h in users[u]):
                if(h != "is_churn" and (users[u][h] == "NaN" or users[u][h] == "nan")):
                    users[u][h] = 0.
                if(users[u][h] == "male"):
                    users[u][h] = 0.
                if(users[u][h] == "female"):
                    users[u][h] = 1.
            else:
                users[u][h] = 0.
        dataset.loc[u] = pd.Series(users[u])
    return dataset

def getTestUsers():
    sample = read_csv("sample_submission_v2.csv", header=None)
    return list(sample[0])

print("Loading data ...")
dataset = gather_dataset()
train = dataset[dataset["is_churn"] == 0 or dataset["is_churn"] == 1]
test = dataset[not(dataset["is_churn"] == 0 or dataset["is_churn"] == 1)]
print("Data loaded.")

print("Analyzing ...")
result = analyse(test, train, 0.7, 'XGBOOST')
print("Exporting result in csv ...")
export_csv(result)
print("Result exported.")

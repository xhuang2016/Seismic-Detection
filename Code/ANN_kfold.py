#!/usr/bin/env python
"""
====================================
k-fold Cross Validation for ANN
====================================
"""
import pandas as pd
import numpy as np
import math
import os
import glob
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

# #=========================== Process EQ data ===================================
## Location of EQ data
work_path = 'EQ'

## Location to save the result
write_path = 'result/'

###======Settings for different test cases======###
SamplingRate = 100 # need to be changed, 25/50/100
Duration = 10 # need to be changed, 2/4/10
###=============================================###

WindowSize = 2 * SamplingRate
original_SamplingRate = 100
rate = original_SamplingRate/SamplingRate

qry = work_path + '/*.txt'
files = glob.glob(qry)
n_events = len(files)
EQ_features = []
for fn in files:
    ## Load EQ data
    data = pd.read_csv(fn, sep=' ',names=['X','Y','Z'])
    ## Down-sampling
    data = data.iloc[0::int(rate)]
    data = data.reset_index(drop=True)
    ## Find the peak of x-axis component
    X = data['X']
    X_peak = np.where(X == np.max(X))[0][0]
    ## Select earthquake duration
    start = X_peak - int(SamplingRate)
    end = X_peak + int(SamplingRate) * (Duration - 1)
    df = data[start:end]
    df = df.reset_index(drop=True)

    X_tg = df['X']
    Y_tg = df['Y']
    Z_tg = df['Z']

    if len(X_tg) == SamplingRate*Duration:
        ## Compute vector sum of three-component
        VS = pow(pow(X_tg, 2) + pow(Y_tg, 2) + pow(Z_tg, 2), 1/2)

        ## 2 sec sliding window with 1 sec overlap
        for j in np.arange(0, len(X)-SamplingRate, SamplingRate):
            train = VS[j:j+WindowSize]
            if len(train) == WindowSize:
                ## IQR
                Q75, Q25 = np.percentile(train, [75 ,25])
                IQR = Q75 - Q25
                ## ZC
                ZCx=0
                ZCy=0
                ZCz=0
                ## CAV
                CAV=0
                for i in range(j,j+WindowSize-1):
                    if ((X_tg[i]<0) != (X_tg[i+1]<0)):
                        ZCx += 1
                    if ((Y_tg[i]<0) != (Y_tg[i+1]<0)):
                        ZCy += 1
                    if ((Z_tg[i]<0) != (Z_tg[i+1]<0)):
                        ZCz += 1
                    amp = (VS[i] + VS[i+1]) / 2.0
                    CAV += amp/SamplingRate
                ZC = max(ZCx,ZCy,ZCz)
                EQ_feature = [IQR, ZC, CAV]
                EQ_features.append(EQ_feature)
EQ_features = np.reshape(EQ_features, (int(len(EQ_features)/(Duration-1)), Duration-1, 3))

# #=========================== Process HumanActivity data ===================================
## Location of Non-Earthquake data
work_path2 = 'NonEQ'

HA_features = []
for root, dirs, files in os.walk(work_path2):
    for folders in dirs:
        qry2 = work_path2 + '/'+ folders + '/*.csv'
        files2 = glob.glob(qry2)
        for fn2 in files2:
            ## Load Non-EQ data
            data = pd.read_csv(fn2, header=0)
            ## Down-sampling
            df = data.iloc[0::int(rate)]
            df = df.reset_index(drop=True)

            X = df['x']
            Y = df['y']
            Z = df['z']

            ## Move to ground by subtracting the mean, and convert [m/s^2] to [g]
            X_tg = (X - np.mean(X))/9.80665
            Y_tg = (Y - np.mean(Y))/9.80665
            Z_tg = (Z - np.mean(Z))/9.80665

            ## Compute vector sum of three-component
            VS = pow(pow(X_tg, 2) + pow(Y_tg, 2) + pow(Z_tg, 2), 1/2)

            ## 2 sec sliding window with 1 sec overlap
            for j in np.arange(0, len(X)-SamplingRate, SamplingRate):
                train = VS[j:j+WindowSize]
                if len(train) == WindowSize:
                    ## IQR
                    Q75, Q25 = np.percentile(train, [75 ,25])
                    IQR = Q75 - Q25
                    ## ZC
                    ZCx=0
                    ZCy=0
                    ZCz=0
                    ## CAV
                    CAV=0
                    for i in range(j,j+WindowSize-1):
                        if ((X_tg[i]<0) != (X_tg[i+1]<0)):
                            ZCx += 1
                        if ((Y_tg[i]<0) != (Y_tg[i+1]<0)):
                            ZCy += 1
                        if ((Z_tg[i]<0) != (Z_tg[i+1]<0)):
                            ZCz += 1
                        amp = (VS[i] + VS[i+1]) / 2.0
                        CAV += amp/SamplingRate
                    ZC = max(ZCx,ZCy,ZCz)
                    HA_feature = [IQR, ZC, CAV]
                    HA_features.append(HA_feature)
HA_features = np.reshape(HA_features, (len(HA_features), 1, 3))

# #=========================== k-fold CV ===================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
i = 0
for (train_index, test_index), (train_index2, test_index2) in zip(kf.split(EQ_features), kf.split(HA_features)):
    EQ_train, EQ_test = EQ_features[train_index], EQ_features[test_index]
    HA_train, HA_test = HA_features[train_index2], HA_features[test_index2]

    # #=========================== k-means ===================================
    EQ_train = np.reshape(EQ_train, (len(EQ_train)*(Duration-1), 3))
    EQ_train_y = np.ones(len(EQ_train))

    HA_train = np.reshape(HA_train, (len(HA_train), 3))

    if len(EQ_train) < len(HA_train):
        ## k-means clustering to balance the dataset
        ## k-means clustering only runs when # of EQ instances < # of Human Activity instances
        kmeans = KMeans(n_clusters=len(EQ_train), random_state=42).fit(HA_train)

        HA_train_centroid = kmeans.cluster_centers_
        HA_train_y = np.zeros(len(HA_train_centroid))

        ANN_train_X = np.vstack((EQ_train, HA_train_centroid))
        ANN_train_y = np.hstack((EQ_train_y, HA_train_y))
    else:
        HA_train_y = np.zeros(len(HA_train))

        ANN_train_X = np.vstack((EQ_train, HA_train))
        ANN_train_y = np.hstack((EQ_train_y, HA_train_y))


    EQ_test = np.reshape(EQ_test, (len(EQ_test)*(Duration-1), 3))
    EQ_test_y = np.ones(len(EQ_test))

    HA_test = np.reshape(HA_test, (len(HA_test), 3))
    HA_test_y = np.zeros(len(HA_test))

    ANN_test_X = np.vstack((EQ_test, HA_test))
    ANN_test_y = np.hstack((EQ_test_y, HA_test_y))

    # #=========================== ANN ===================================
    ## Feature scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    ANN_train_X = min_max_scaler.fit_transform(ANN_train_X)
    ANN_test_X = min_max_scaler.transform(ANN_test_X)

    ## ANN Training
    mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', solver='sgd', alpha=0, max_iter=10000, random_state=42, learning_rate_init=0.2)
    mlp.fit(ANN_train_X, ANN_train_y.ravel())
    ## ANN Testing
    y_prob = mlp.predict_proba(ANN_test_X)

    ## Save the output prediction probability
    result = np.concatenate((ANN_test_y.reshape(-1, 1), y_prob), axis=1)
    df_result = pd.DataFrame(result, columns=['labels','prob_0','prob_1'])
    df_result.to_csv(write_path + 'ANN_%s'%SamplingRate + 'Hz_%s'%Duration +'s_fold_%s.csv'% i, index=False)
    i += 1

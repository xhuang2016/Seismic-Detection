#!/usr/bin/env python
"""
====================================
Implementation of CRNN
====================================
"""
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf

# #=========================== Process EQ data ===================================
## Location of EQ data
work_path = 'EQ'

## Location to save the result
write_path = 'result/'

###======Settings for different test cases======###
SamplingRate = 25 # need to be changed, 25/50/100
Duration = 2 # need to be changed, 2/4/10
###============###

WindowSize = 2 * SamplingRate
original_SamplingRate = 100
rate = original_SamplingRate/SamplingRate

qry = work_path + '/*.txt'
files = glob.glob(qry)
X_EQ=[]
Y_EQ=[]
Z_EQ=[]
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
        ## 2 sec sliding window with 1 sec overlap
        for j in np.arange(0, len(X_tg)-SamplingRate, SamplingRate):
            X_batch = X_tg[j:j+WindowSize]
            Y_batch = Y_tg[j:j+WindowSize]
            Z_batch = Z_tg[j:j+WindowSize]
            if len(X_batch) == WindowSize:
                X_EQ.append(X_batch.values)
                Y_EQ.append(Y_batch.values)
                Z_EQ.append(Z_batch.values)

X_EQ = np.asarray(X_EQ)
Y_EQ = np.asarray(Y_EQ)
Z_EQ = np.asarray(Z_EQ)

X_EQ = X_EQ.reshape(int(len(X_EQ)/(Duration-1)), Duration-1, WindowSize, 1)
Y_EQ = Y_EQ.reshape(int(len(Y_EQ)/(Duration-1)), Duration-1, WindowSize, 1)
Z_EQ = Z_EQ.reshape(int(len(Z_EQ)/(Duration-1)), Duration-1, WindowSize, 1)

# #=========================== Process Non-Earthquake data ===================================
## Location of Non-Earthquake data
work_path2 = 'NonEQ'

X_HA=[]
Y_HA=[]
Z_HA=[]
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

            ## 2 sec sliding window with 1 sec overlap
            for j in np.arange(0, len(X)-SamplingRate, SamplingRate):
                X_batch = X_tg[j:j+WindowSize]
                Y_batch = Y_tg[j:j+WindowSize]
                Z_batch = Z_tg[j:j+WindowSize]
                if len(X_batch) == WindowSize:
                    X_HA.append(X_batch.values)
                    Y_HA.append(Y_batch.values)
                    Z_HA.append(Z_batch.values)

X_HA = np.asarray(X_HA)
Y_HA = np.asarray(Y_HA)
Z_HA = np.asarray(Z_HA)

X_HA = X_HA.reshape(X_HA.shape[0],X_HA.shape[1],1)
Y_HA = Y_HA.reshape(Y_HA.shape[0],Y_HA.shape[1],1)
Z_HA = Z_HA.reshape(Z_HA.shape[0],Z_HA.shape[1],1)

### Split the data into training set and testing set
indices = np.arange(len(X_EQ))
X_EQ_train, X_EQ_test, train_index, test_index = train_test_split(X_EQ, indices, test_size=0.3, random_state=42)
Y_EQ_train, Y_EQ_test = Y_EQ[train_index], Y_EQ[test_index]
Z_EQ_train, Z_EQ_test = Z_EQ[train_index], Z_EQ[test_index]

X_EQ_train = X_EQ_train.reshape(X_EQ_train.shape[0]*X_EQ_train.shape[1], X_EQ_train.shape[2], 1)
Y_EQ_train = Y_EQ_train.reshape(Y_EQ_train.shape[0]*Y_EQ_train.shape[1], Y_EQ_train.shape[2], 1)
Z_EQ_train = Z_EQ_train.reshape(Z_EQ_train.shape[0]*Z_EQ_train.shape[1], Z_EQ_train.shape[2], 1)

X_EQ_test = X_EQ_test.reshape(X_EQ_test.shape[0]*X_EQ_test.shape[1], X_EQ_test.shape[2], 1)
Y_EQ_test = Y_EQ_test.reshape(Y_EQ_test.shape[0]*Y_EQ_test.shape[1], Y_EQ_test.shape[2], 1)
Z_EQ_test = Z_EQ_test.reshape(Z_EQ_test.shape[0]*Z_EQ_test.shape[1], Z_EQ_test.shape[2], 1)


indices2 = np.arange(len(X_HA))
X_HA_train, X_HA_test, train_index2, test_index2 = train_test_split(X_HA, indices2, test_size=0.3, random_state=42)
Y_HA_train, Y_HA_test = Y_HA[train_index2], Y_HA[test_index2]
Z_HA_train, Z_HA_test = Z_HA[train_index2], Z_HA[test_index2]

# #=========================== CRNN ===================================
### CRNN Training
EQ_X_train = np.dstack((X_EQ_train, Y_EQ_train, Z_EQ_train))
HA_X_train = np.dstack((X_HA_train, Y_HA_train, Z_HA_train))

EQ_y_train = np.ones(len(X_EQ_train))
HA_y_train = np.zeros(len(X_HA_train))

X_train = np.vstack((EQ_X_train, HA_X_train))
y_train = np.hstack((EQ_y_train, HA_y_train)).reshape(-1,1)

ratio = float(len(HA_y_train)/len(EQ_y_train))
class_weights = {0: 1., 1: ratio}

verbose, epochs, batch_size = 2, 100, 256
n_features, n_outputs = X_train.shape[2], y_train.shape[1]
n_steps, n_length = 2, SamplingRate

X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
model.add(tf.keras.layers.SimpleRNN(100))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights, verbose=verbose)
model.save(write_path + 'CRNN_%s'%SamplingRate + 'Hz_%s'%Duration +'s.h5')

### CRNN Testing
EQ_X_test = np.dstack((X_EQ_test, Y_EQ_test, Z_EQ_test))
HA_X_test = np.dstack((X_HA_test, Y_HA_test, Z_HA_test))

EQ_y_test = np.ones(len(X_EQ_test))
HA_y_test = np.zeros(len(X_HA_test))

X_test = np.vstack((EQ_X_test, HA_X_test))
y_test = np.hstack((EQ_y_test, HA_y_test)).reshape(-1,1)

X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))

y_prob = model.predict(X_test).ravel()

## Save the ouput prediction probability
result = np.concatenate((y_test, y_prob.reshape(-1, 1)),axis=1)
df_result = pd.DataFrame(result, columns=['labels','prob_1'])
df_result.to_csv(write_path + 'CRNN_%s'%SamplingRate + 'Hz_%s'%Duration +'s.csv', index=False)

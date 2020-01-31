#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import glob
# from sklearn.model_selection import KFold
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.layers import TimeDistributed
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D
# from keras.layers import SimpleRNN
import tensorflow as tf

#=========================== Process Earthquake (EQ) data ===================================
work_path = 'windows/10s' # location of EQ data
write_path = 'model/' # location of output

SamplingRate = 50 # target sampling rate, need to be changed, 25/50/100
Duration = 10 # duration of the EQ, need to be changed, 2/4/10

original_SamplingRate = 100 # original sampling rate of the dataset
WindowSize = 2 * SamplingRate # sliding window size
rate = original_SamplingRate/SamplingRate # down-sampling ratio

qry = work_path + '/*.csv'
files = glob.glob(qry)
n_events = len(files)
X_EQ=[]
Y_EQ=[]
Z_EQ=[]
for fn in files:
    data = pd.read_csv(fn, header=0)
    # data = pd.read_csv(fn, sep=' ', names=['X','Y','Z'])
    df = data.iloc[0::int(rate)]
    df = df.reset_index(drop=True)
    # X = data['X']
    # X_peak = np.where(X == np.max(X))[0][0] # find the peak
    # start = X_peak - int(SamplingRate)
    # end = X_peak + int(SamplingRate) * (Duration - 1)
    # df = data[start:end]
    # df = df.reset_index(drop=True)

    X_tg = df['X']
    Y_tg = df['Y']
    Z_tg = df['Z']

    # 2 sec sliding window with 1 sec overlap
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

# reshape the time-series data
X_EQ = X_EQ.reshape(X_EQ.shape[0],X_EQ.shape[1],1)
Y_EQ = Y_EQ.reshape(Y_EQ.shape[0],Y_EQ.shape[1],1)
Z_EQ = Z_EQ.reshape(Z_EQ.shape[0],Z_EQ.shape[1],1)

#=========================== Process Human Activity (HA) data ===================================
work_path2 = 'human' # location of HA data

X_HA=[]
Y_HA=[]
Z_HA=[]
for root, dirs, files in os.walk(work_path2):
    for folders in dirs:
        qry2 = work_path2 + '/'+ folders + '/*.csv'
        files2 = glob.glob(qry2)
        for fn2 in files2:
            data = pd.read_csv(fn2, header=0)
            df = data.iloc[0::int(rate)]
            df = df.reset_index(drop=True)

            X = df['x']
            Y = df['y']
            Z = df['z']

            X_tg = (X - np.mean(X))/9.80665
            Y_tg = (Y - np.mean(Y))/9.80665
            Z_tg = (Z - np.mean(Z))/9.80665

            # 2 sec sliding window with 1 sec overlap
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


# X_EQ_train, X_EQ_test = X_EQ[train_index], X_EQ[test_index]
# Y_EQ_train, Y_EQ_test = Y_EQ[train_index], Y_EQ[test_index]
# Z_EQ_train, Z_EQ_test = Z_EQ[train_index], Z_EQ[test_index]
#
# X_HA_train, X_HA_test = X_HA[train_index2], X_HA[test_index2]
# Y_HA_train, Y_HA_test = Y_HA[train_index2], Y_HA[test_index2]
# Z_HA_train, Z_HA_test = Z_HA[train_index2], Z_HA[test_index2]

#=========================== CRNN ===================================
## CRNN Training
EQ_X_train = np.dstack((X_EQ, Y_EQ, Z_EQ))
HA_X_train = np.dstack((X_HA, Y_HA, Z_HA))

EQ_y_train = np.ones(len(X_EQ))
HA_y_train = np.zeros(len(X_HA))

X_train = np.vstack((EQ_X_train, HA_X_train))
y_train = np.hstack((EQ_y_train, HA_y_train)).reshape(-1,1)

ratio = float(len(HA_y_train)/len(EQ_y_train))
class_weights = {0: 1., 1: ratio}

verbose, epochs, batch_size = 2, 100, 256
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
n_steps, n_length = 2, SamplingRate

X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))

# model = Sequential()
# model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
# model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
# model.add(TimeDistributed(Dropout(0.5)))
# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model.add(TimeDistributed(Flatten()))
# model.add(SimpleRNN(100))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(n_outputs, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights, verbose=verbose)
# model.save('model/CRNN_2s_Keras2.h5')

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
model.save('model/CRNN_50hz_10s_Keras2.h5')

# ## CRNN Testing
# EQ_X_test = np.dstack((X_EQ_test, Y_EQ_test, Z_EQ_test))
# HA_X_test = np.dstack((X_HA_test, Y_HA_test, Z_HA_test))
#
# EQ_y_test = np.ones(len(X_EQ_test))
# HA_y_test = np.zeros(len(X_HA_test))
#
# X_test = np.vstack((EQ_X_test, HA_X_test))
# y_test = np.hstack((EQ_y_test, HA_y_test)).reshape(-1,1)
#
# X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
#
# y_prob = model.predict(X_test).ravel()
# ## save prediction probability
# result = np.concatenate((y_test, y_prob.reshape(-1, 1)),axis=1)
# df_result = pd.DataFrame(result,columns=['labels','prob_1'])
# df_result.to_csv(write_path + 'CRNN_%s'%SamplingRate + 'Hz_%s'%Duration +'s_pred_fold_%s.csv'% i,index=False)
# i += 1



# # #=========================== k-fold CV ===================================
# kf = KFold(n_splits=10, shuffle=True, random_state=42) # 10-fold CV
# i = 0
# for (train_index, test_index), (train_index2, test_index2) in zip(kf.split(X_EQ), kf.split(X_HA)):
#
#     X_EQ_train, X_EQ_test = X_EQ[train_index], X_EQ[test_index]
#     Y_EQ_train, Y_EQ_test = Y_EQ[train_index], Y_EQ[test_index]
#     Z_EQ_train, Z_EQ_test = Z_EQ[train_index], Z_EQ[test_index]
#
#     X_HA_train, X_HA_test = X_HA[train_index2], X_HA[test_index2]
#     Y_HA_train, Y_HA_test = Y_HA[train_index2], Y_HA[test_index2]
#     Z_HA_train, Z_HA_test = Z_HA[train_index2], Z_HA[test_index2]
#
#     #=========================== CRNN ===================================
#     ## CRNN Training
#     EQ_X_train = np.dstack((X_EQ_train, Y_EQ_train, Z_EQ_train))
#     HA_X_train = np.dstack((X_HA_train, Y_HA_train, Z_HA_train))
#
#     EQ_y_train = np.ones(len(X_EQ_train))
#     HA_y_train = np.zeros(len(X_HA_train))
#
#     X_train = np.vstack((EQ_X_train, HA_X_train))
#     y_train = np.hstack((EQ_y_train, HA_y_train)).reshape(-1,1)
#
#     ratio = float(len(HA_y_train)/len(EQ_y_train))
#     class_weights = {0: 1., 1: ratio}
#
#     verbose, epochs, batch_size = 0, 100, 256
#     n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
#     n_steps, n_length = 2, SamplingRate
#
#     X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
#
#     model = Sequential()
#     model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
#     model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
#     # model.add(TimeDistributed(Dropout(0.5)))
#     model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
#     model.add(TimeDistributed(Flatten()))
#     model.add(SimpleRNN(100))
#     model.add(Dropout(0.5))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights, verbose=verbose)
#
#     ## CRNN Testing
#     EQ_X_test = np.dstack((X_EQ_test, Y_EQ_test, Z_EQ_test))
#     HA_X_test = np.dstack((X_HA_test, Y_HA_test, Z_HA_test))
#
#     EQ_y_test = np.ones(len(X_EQ_test))
#     HA_y_test = np.zeros(len(X_HA_test))
#
#     X_test = np.vstack((EQ_X_test, HA_X_test))
#     y_test = np.hstack((EQ_y_test, HA_y_test)).reshape(-1,1)
#
#     X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
#
#     y_prob = model.predict(X_test).ravel()
#     ## save prediction probability
#     result = np.concatenate((y_test, y_prob.reshape(-1, 1)),axis=1)
#     df_result = pd.DataFrame(result,columns=['labels','prob_1'])
#     df_result.to_csv(write_path + 'CRNN_%s'%SamplingRate + 'Hz_%s'%Duration +'s_pred_fold_%s.csv'% i,index=False)
#     i += 1

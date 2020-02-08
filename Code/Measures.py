#!/usr/bin/env python
"""
====================================
Calculation of the Performance Measures
====================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

## Load the result, please change the file name
data = pd.read_csv('result/CRNN_25hz_10s.csv', header=0)

y_prob = data['prob_1']
y_test = data['labels']

## Set Probability Threshold
P_Threshold = 0.5
## Classification based on the Probability Threshold
y_pred = np.where(y_prob > P_Threshold, 1, 0)
## Calculate Confusion Matrix
confusionmatrix = confusion_matrix(y_test, y_pred)

## Calculate AUROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
## Calculate AUPR
precision, recall, thresholds2 = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)


## Print Performance Measures
## AUROC
print('AUROC:', roc_auc)
## AUPR
print('AUPRC:', pr_auc)
## Performance Measures with a Probability Threshold of 0.5
print('Threshold 0.5:')
## Confusion Matrix
print(confusionmatrix)
## Classification Report
print(classification_report(y_test, y_pred, digits=4))
## Accuracy
print('Accuracy of the model on test set: {:.8f}'.format((confusionmatrix[0][0]+confusionmatrix[1][1])/np.sum(confusionmatrix)))
## Recall
print('Recall',recall_score(y_test, y_pred, average='binary'))
## Precision
print('Precision',precision_score(y_test, y_pred, average='binary'))
## False-alarm rate
print('False-alarm rate:', confusionmatrix[0][1]/(confusionmatrix[0][0]+confusionmatrix[0][1]))
# ## F1 score
# print('F1:',f1_score(y_test, y_pred, average='binary'))

## Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue',linewidth=3)
plt.tick_params(direction='in',length=12)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.text(0.8, 0.1, 'AUROC: %0.4f'% roc_auc)
plt.tight_layout()
# plt.savefig('result/CRNN_25hz_10s_ROC.png', format='png', dpi=100)
plt.show()

## Plot PR curve
plt.figure()
plt.plot(recall, precision, color='blue',linewidth=3)
plt.tick_params(direction='in',length=12)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.text(0.8, 0.1, 'AUPR: %0.4f'% pr_auc)
plt.tight_layout()
# plt.savefig('result/CRNN_25hz_10s_PR.png', format='png', dpi=100)
plt.show()

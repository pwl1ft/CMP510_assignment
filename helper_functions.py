# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:19:33 2023

@author: User
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix


separator = '-'

def substringCategory(row):
    return row.split(separator, 1)[0]

def visualiseFeatureImportance(regressor, rawData):
    # # Visual
    # # Plot bar chart of feature importance
    importance = regressor.feature_importances_
    print(importance)#extract feature importance data
    features = list(rawData) #get list of features
    d = {'feature': features, 'importance': list(importance)}
    dfGraph = pd.DataFrame(data=d) #create dataframe
    dfGraph.sort_values(by=['importance'], inplace=True, ascending=False) #sort features by importance
    plt.bar(range(0,len(importance)), dfGraph['importance'], color='green')
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(ticks=range(0,len(importance)), labels=dfGraph['feature'], rotation = 90)
    plt.title("Feature Importance")
    return plt.show()


def createConfusionMatrix(y_test_updated, y_pred_updated): 
    mat = confusion_matrix(y_test_updated, y_pred_updated)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    return mat;

def plotRocRandomForest(fpr, tpr, roc_auc, name):
    plt.figure()
    plt.plot(fpr, tpr, label='AUC = %0.6f' % (roc_auc))
    plt.xlabel('Random forest: False Positive Rate (1-specificity)')
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('Random Forest ROC curve (for detecting %s)' % (name))
    plt.legend(loc="lower right", prop={'size': 'small'})
    return plt.show()

def plotPRRandomForest(recall, precision, auc_pr, name):
    plt.figure()
    plt.plot(recall, precision, label='AUC = %0.6f' % (auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('Random forest Precision Recall curve (for detecting %s)' % (name))
    plt.legend(loc="lower right", prop={'size': 'small'})
    return plt.show()

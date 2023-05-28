# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:46:44 2023

@author: User
"""

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import helper_functions
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns; sns.set()
   
data_dir=""
raw_data_filename_training = data_dir + "CMP510_training_dataset2.csv"
raw_data_filename_test = data_dir + "CMP510_testing_dataset2.csv"

rawData = pd.read_csv(raw_data_filename_training)
rawData= rawData.iloc[:,1:57]
rawData['Class'], protocols = pd.factorize(rawData['Class'])

dataTraining = pd.read_csv(raw_data_filename_training)
dataTest = pd.read_csv(raw_data_filename_test)

# Remove duplicates
dataTraining = dataTraining.drop_duplicates()
dataTest = dataTest.drop_duplicates()

# Substring category to remove everything after and including '-'
dataTraining['Category'] = dataTraining['Category'].apply(helper_functions.substringCategory)
dataTest['Category'] = dataTest['Category'].apply(helper_functions.substringCategory)

print("Category: " + dataTraining['Category'].unique())

# Factorize column "Class"
dataTraining['Class'], protocols= pd.factorize(dataTraining['Class'])
dataTest['Class'], protocols= pd.factorize(dataTest['Class'])

# factorise column "Category"
dataTraining['Category'], protocols= pd.factorize(dataTraining['Category'])
dataTest['Category'], protocols= pd.factorize(dataTest['Category'])

# Arrange training/testing data into X and Y values
X_train = dataTraining.iloc[:, 1:57].values
y_train = dataTraining.iloc[:, 0].values
X_test = dataTest.iloc[:, 1:57].values
y_test = dataTest.iloc[:, 0].values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


helper_functions.visualiseFeatureImportance(regressor, rawData)

# remove the columns that have no bearing on the results
dataTrainingUpdated = dataTraining.drop(labels=["svcscan.fs_drivers", "svcscan.interactive_process_services", 
                                          "handles.nport", "pslist.nprocs64bit"], axis=1)

dataTestUpdated = dataTest.drop(labels=["svcscan.fs_drivers", "svcscan.interactive_process_services", 
                                          "handles.nport", "pslist.nprocs64bit"], axis=1)


# update the training data and run again without unnecessary columns
X_train_updated = dataTrainingUpdated.iloc[:, 1:].values
y_train_updated = dataTrainingUpdated.iloc[:, 0].values
X_test_updated = dataTestUpdated.iloc[:, 1:].values
y_test_updated = dataTestUpdated.iloc[:, 0].values

sc = StandardScaler()
X_train_updated = sc.fit_transform(X_train)
X_test_updated = sc.transform(X_test)

# update the model to use new data
modelUpdated = RandomForestClassifier(n_estimators=100, random_state=42)
modelUpdated.fit(X_train_updated, y_train_updated)
y_pred_updated = modelUpdated.predict(X_test_updated)
y_probabilities_rf = modelUpdated.predict_proba(X_test_updated)

print ("Score after removing columns: ", modelUpdated.score(X_train_updated, y_train_updated))

print('After removing columns: Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_updated))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_updated, y_pred_updated))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_updated, y_pred_updated)))

# get the classification report
print(metrics.classification_report(y_pred_updated, y_test_updated))

# create the confusion matrix
helper_functions.createConfusionMatrix(y_test_updated, y_pred_updated)

# Plot Receiver Operating Characteristic (ROC) and Precision-Recall Curves
# Convert classification output to 0 for normal and 1 for attacks of any kind
y_test = []
y_score = []

y_test_ransomware = []
y_score_ransomware = []

y_test_spyware = []
y_score_spyware = []

y_test_trojan = []
y_score_trojan = []

for n in range(len(y_test_updated)):
    label = y_test_updated[n]
    y_score.append(sum(y_probabilities_rf[n,:])-y_probabilities_rf[n,0]) #sum probabilities of all attack categories excluding normal
    # for each row, take sum of all columns (1) and minus probabilities of 0 (benign)
    if y_pred_updated[n] == 0 and label == 0:
        y_test.append(0)
    else:
        y_test.append(1)
        
# ransomware only        
for n in range(len(y_test_updated)):
    label = y_test_updated[n]
    y_score_ransomware.append(sum(y_probabilities_rf[n,:])-y_probabilities_rf[n,1]) #sum probabilities of all attack categories excluding normal
    # for each row, take sum of all columns (1) and minus probabilities of 0 (benign)
    if y_pred_updated[n] == 1 and label == 1:
         y_test_ransomware.append(0)
    else:
        y_test_ransomware.append(1)    
        
# spyware only       
for n in range(len(y_test_updated)):
    label = y_test_updated[n]
    y_score_spyware.append(sum(y_probabilities_rf[n,:])-y_probabilities_rf[n,2]) #sum probabilities of all attack categories excluding normal
    # for each row, take sum of all columns (1) and minus probabilities of 0 (benign)
    if y_pred_updated[n] == 2 and label == 2:
         y_test_spyware.append(0)
    else:
        y_test_spyware.append(1) 

# trojan only       
for n in range(len(y_test_updated)):
    label = y_test_updated[n]
    y_score_trojan.append(sum(y_probabilities_rf[n,:])-y_probabilities_rf[n,3]) #sum probabilities of all attack categories excluding normal
    # for each row, take sum of all columns (1) and minus probabilities of 0 (benign)
    if y_pred_updated[n] == 3 and label == 3:
         y_test_trojan.append(0)
    else:
        y_test_trojan.append(1)         
   
            
y_test = np.array(y_test)
y_score = np.array(y_score)

y_test_ransomware = np.array(y_test_ransomware)
y_score_ransomware = np.array(y_score_ransomware)

y_test_spyware = np.array(y_test_spyware)
y_score_spyware = np.array(y_score_spyware)

y_test_trojan = np.array(y_test_trojan)
y_score_trojan = np.array(y_score_trojan)

fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
roc_auc = auc(fpr, tpr)
precision, recall, thresholds_precision_recall = precision_recall_curve(y_test, y_score, pos_label=1)
auc_pr = auc(recall, precision)

fpr_rw, tpr_rw, thresholds_rw = roc_curve(y_test_ransomware, y_score_ransomware, pos_label=1)
roc_auc_rw = auc(fpr_rw, tpr_rw)
precision_rw, recall_rw, thresholds_precision_recall_rw = precision_recall_curve(y_test_ransomware, y_score_ransomware, pos_label=1)
auc_pr_rw = auc(recall_rw, precision_rw)

fpr_sw, tpr_sw, thresholds_sw = roc_curve(y_test_spyware, y_score_spyware, pos_label=1)
roc_auc_sw = auc(fpr_sw, tpr_sw)
precision_sw, recall_sw, thresholds_precision_recall_sw = precision_recall_curve(y_test_spyware, y_score_spyware, pos_label=1)
auc_pr_sw = auc(recall_sw, precision_sw)

fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_test_trojan, y_score_trojan, pos_label=1)
roc_auc_tr = auc(fpr_tr, tpr_tr)
precision_tr, recall_tr, thresholds_precision_recall_tr = precision_recall_curve(y_test_trojan, y_score_trojan, pos_label=1)
auc_pr_tr = auc(recall_tr, precision_tr)

helper_functions.plotRocRandomForest(fpr, tpr, roc_auc, "all attacks")
helper_functions.plotRocRandomForest(fpr_rw, tpr_rw, roc_auc_rw, "ransomware")
helper_functions.plotRocRandomForest(fpr_sw, tpr_sw, roc_auc_sw, "spyware")
helper_functions.plotRocRandomForest(fpr_tr, tpr_tr, roc_auc_tr, "trojan")
helper_functions.plotPRRandomForest(recall, precision, auc_pr, "all attacks")
helper_functions.plotPRRandomForest(recall_rw, precision_rw, auc_pr_rw, "ransomware")
helper_functions.plotPRRandomForest(recall_sw, precision_sw, auc_pr_sw, "spyware")
helper_functions.plotPRRandomForest(recall_tr, precision_tr, auc_pr_tr, "trojan")

print("ROC AUC all attacks: " + str(roc_auc))
print("ROC AUC ransomware: " + str(roc_auc_rw))
print("ROC AUC spyware: " + str(roc_auc_sw))
print("ROC AUC trojan: " + str(roc_auc_tr))
print("PR AUC all attacks: " + str(auc_pr))
print("PR AUC ransomware: " + str(auc_pr_rw))
print("PR AUC spyware: " + str(auc_pr_sw))
print("PR AUC trojan: " + str(auc_pr_tr))

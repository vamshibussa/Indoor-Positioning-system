# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 11:37:32 2018

@author: Adluri Saivihar Raj
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("trainingData.csv",header = 0)

features = np.asarray(dataset.iloc[:,0:520])
features[features == 100] = -110
features = (features - features.mean()) / features.var()

labels = np.asarray(dataset["BUILDINGID"].map(str) + dataset["FLOOR"].map(str))
labels = np.asarray(pd.get_dummies(labels))


# #### Dividing UJIndoorLoc training data set into training and validation set



train_val_split = np.random.rand(len(features)) < 0.70
train_x = features[train_val_split]
train_y = labels[train_val_split]
val_x = features[~train_val_split]
val_y = labels[~train_val_split]


#### Using UJIndoorLoc validation data set as testing set



test_dataset = pd.read_csv("validationData.csv",header = 0)
#print(test_dataset)
test_features = np.asarray(test_dataset.iloc[:,0:520])
#print(test_features)
test_features[test_features == 100] = -110
#print(test_features)
test_features = (test_features - test_features.mean()) / test_features.var()
#print(test_features)
test_labels = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))
#print(test_labels)
test_labels = np.asarray(pd.get_dummies(test_labels))

##############################################################################################################






for K in range(25):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
    neigh.fit(train_x,train_y) 
    y_pred = neigh.predict(val_x)
    print(y_pred[0])
    print(len(y_pred[0]))
    print ("Accuracy is ", accuracy_score(val_y,y_pred)*100,"% for K-Value:",K_value)









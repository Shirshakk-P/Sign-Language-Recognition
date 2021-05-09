#!/usr/bin/env python
# coding: utf-8

# Sign Language Recognition using Random Forests

#Importing the Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split


data=pd.read_csv('/home/shark_p/Downloads/all/ASL_train.csv');

#Dataset Info:
df_x = data.iloc[:,1:];
df_y = data.iloc[:,0];


x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4);


rf = RandomForestClassifier(n_estimators=100);
rf.fit(x_train,y_train);


pred=rf.predict(x_test);

count=0;
s = y_test.values;
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count + 1;

#Confusion Matrix:
plot_confusion_matrix(rf, x_test, y_test, cmap=plt.cm.CMRmap)
plt.figure(figsize=(48, 48))
plt.show()

#Accuracy of the Random Forest Model:        
print((count/len(pred)) *100);

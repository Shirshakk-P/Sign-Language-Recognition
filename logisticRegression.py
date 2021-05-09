#!/usr/bin/env python
# coding: utf-8


# Sign Language Recognition using Logistic Regression

#Importing the Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



df=pd.read_csv('/home/shark_p/Downloads/all/ASL_train.csv')
print(df.shape)
df.head()



X=df.values[0:,1:]
Y = df.values[0:,0]


#Dataset Vizualization of random value:
sample = X[1]
plt.imshow(sample.reshape((28,28)))



Y.shape


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.30)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0


#Building the Logistic Regression Model:
LR = LogisticRegression(penalty='none', 
                         tol=0.1, solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, Y_train)



Y_pred=LR.predict(X_test_scaled)
Y_test
Y_pred



#Building the Confusion Matrix 
result = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(result)



#Classification report
result1 = classification_report(Y_test, Y_pred)
print("\nClassification Report:")
print (result1)



# Accuracy of the Logistic Regression Model:
result2 = accuracy_score(Y_test, Y_pred)
print("\nAccuracy:",result2)


#Confusion Matrix
plot_confusion_matrix(LR, X_test, Y_test, cmap=plt.cm.CMRmap)
plt.figure(figsize=(48, 48))
plt.show()

#!/usr/bin/env python
# coding: utf-8


# Sign Language Recognition using Support Vector Machine

#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('/home/shark_p/Downloads/all/ASL_train.csv')
print(df.shape)
df.head()


X = df.iloc[0:,1:].values
X = X/225
X


Y = df.iloc[0:,0].values
Y


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 50)



scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


#Building the Support Vector Machine:
supportvector = svm.SVC(gamma = 0.001, C = 1000, decision_function_shape='ovo', random_state = 100)
supportvector.fit(X_train, y_train)


#Confusion Matrix:
plot_confusion_matrix(supportvector, X_test, y_test, cmap=plt.cm.CMRmap)
plt.figure(figsize=(24, 24))
plt.show()

#Accuracy of the SVM Model:
support_pred = supportvector.predict(X_test)
support_accuracy = accuracy_score(y_test, support_pred)
print(support_accuracy)

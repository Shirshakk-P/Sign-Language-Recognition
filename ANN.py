#!/usr/bin/env python
# coding: utf-8

# Sign Language Recognition using ANN

#Importing the Libraries
import pandas as pd
import keras
import tensorflow as tf
import numpy as np


from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten


df = pd.read_csv('/home/shark_p/Downloads/all/ASL_train.csv')
df.head()


X = df.iloc[:,1:]
X = X/225
X


Y = df.iloc[:,0]
Y


Y = Y.astype(int)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state=0)


scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


#Building the ANN Model
model = Sequential(Flatten(input_shape = [28, 28]))
model.add(Dense(300, activation='relu'))
model.add(Dropout(rate= 0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(25, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam',
              metrics='accuracy')
h = model.fit(X_train, y_train, epochs=6, verbose=True)


#Plot for Model Accuracy:
plt.plot(h.history['accuracy'])
plt.title('Model accuracy')
plt.show()

#Acuuracy of the ANN Model:
['accuracy']

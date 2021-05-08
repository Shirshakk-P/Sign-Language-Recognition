#!/usr/bin/env python
# coding: utf-8

# In[40]:


#LOADING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import plot_confusion_matrix
from keras.utils import to_categorical 
from keras import backend as K
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.core import Activation


# In[41]:


#LOADING DATASET
df=pd.read_csv('/home/shark_p/Downloads/all/ASL_train.csv')
print(df.shape)
df.head()


# In[42]:


X=df.values[0:,1:]
Y = df.values[0:,0]

sample = X[1]
plt.imshow(sample.reshape((28,28)))


# In[43]:


Y.shape


# In[44]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0,
                                      test_size=0.25)


# In[45]:


X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0


# In[46]:


lr = LinearRegression()
lr.fit(X_train_scaled,Y_train)


# In[47]:


Y_pred=lr.predict(X_test_scaled)

Y_pred


# In[48]:


accuracy =r2_score(Y_test, Y_pred)
print(accuracy)

print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, Y_pred))


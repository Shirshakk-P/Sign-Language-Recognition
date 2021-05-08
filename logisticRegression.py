#!/usr/bin/env python
# coding: utf-8

# In[50]:


#LOADING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[51]:


#LOADING DATASET
df=pd.read_csv('/home/shark_p/Downloads/all/ASL_train.csv')
print(df.shape)
df.head()


# In[52]:


X=df.values[0:,1:]
Y = df.values[0:,0]

sample = X[1]
plt.imshow(sample.reshape((28,28)))


# In[53]:


Y.shape


# In[54]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1,
                                      test_size=0.25)


# In[56]:


X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0


# In[64]:


LR = LogisticRegression(penalty='none', 
                         tol=0.1, solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, Y_train)


# In[65]:


Y_pred=LR.predict(X_test_scaled)
Y_test
Y_pred


# In[67]:


# Confusion Matrix 
result = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(result)

# Classification report
result1 = classification_report(Y_test, Y_pred)
print("\nClassification Report:")
print (result1)

# Accuracy score
result2 = accuracy_score(Y_test, Y_pred)
print("\nAccuracy:",result2)

cm = pd.crosstab(Y_test, Y_pred, 
                               rownames=['Actual'], colnames=['Predicted'], normalize='index')
p = plt.figure(figsize=(10,10));
p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)


#!/usr/bin/env python
# coding: utf-8

# Sign Language Recognition using CNN

#Importing the Libraries
import os
import numpy as np 
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_df=pd.read_csv('/home/shark_p/Downloads/all/ASL_train.csv')
test_df=pd.read_csv('/home/shark_p/Downloads/all/ASL_test.csv')


train_df.info()

test_df.info()


train_df.describe()

train_df.head(6)



train_label=train_df['label']
train_label.head()
trainset=train_df.drop(['label'],axis=1)
trainset.head()


X_train = trainset.values
X_train = trainset.values.reshape(-1,28,28,1)
print(X_train.shape)


test_label=test_df['label']
X_test=test_df.drop(['label'],axis=1)
print(X_test.shape)
X_test.head()



from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train=lb.fit_transform(train_label)
y_test=lb.fit_transform(test_label)
y_train


X_test=X_test.values.reshape(-1,28,28,1)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)



train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 0,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

X_test=X_test/255


#Dataset Visualization of Random Characters:
fig,axe=plt.subplots(2,2)
fig.suptitle('Preview of dataset')
axe[0,0].imshow(X_train[0].reshape(28,28),cmap='gray')
axe[0,0].set_title('label: 3  letter: C')
axe[0,1].imshow(X_train[1].reshape(28,28),cmap='gray')
axe[0,1].set_title('label: 6  letter: F')
axe[1,0].imshow(X_train[2].reshape(28,28),cmap='gray')
axe[1,0].set_title('label: 2  letter: B')
axe[1,1].imshow(X_train[4].reshape(28,28),cmap='gray')
axe[1,1].set_title('label: 13  letter: M')



sns.countplot(train_label)
plt.title("Frequency of each label")


#CNN Model:
# Conv Layer 1 -- UNITS - 128  KERNEL SIZE - 5 * 5   STRIDE LENGTH - 1   ACTIVATION - ReLu
# Conv Layer 2 -- UNITS - 64   KERNEL SIZE - 3 * 3   STRIDE LENGTH - 1   ACTIVATION - ReLu
# Conv Layer 3 -- UNITS - 32   KERNEL SIZE - 2 * 2   STRIDE LENGTH - 1   ACTIVATION - ReLu
# MaxPool Layer 1 -- MAX POOL WINDOW - 3 * 3   STRIDE - 2
# MaxPool Layer 2 -- MAX POOL WINDOW - 2 * 2   STRIDE - 2
# MaxPool Layer 3 -- MAX POOL WINDOW - 2 * 2   STRIDE - 2

model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))      
model.add(Flatten())



model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


#Model Training:
model.fit(train_datagen.flow(X_train,y_train,batch_size=200), epochs = 35,
          validation_data=(X_test,y_test), shuffle=1)



(ls,acc)=model.evaluate(x=X_test,y=y_test)

#Accuracy of the CNN Model:
print('Model Accuracy = {}%'.format(acc*100))


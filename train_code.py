#https://github.com/abhishek-v
import pandas as pd
import matplotlib
import numpy as np
import h5py as h5py
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Convolution2D,Dense,Dropout,Flatten
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


#displaying the image
# a=np.array(df.iloc[[3244]])
# size=int(a.shape[1]**(1/2))
# a.resize(size,size)
# plt.imshow(a)
# plt.show()

#reading input
train=pd.read_csv("train.csv")

train_labels=train["label"]
train=train.drop("label",axis=1)

train=train.values.astype('float32')
train.resize(train.shape[0],28,28,1)

#reading output
test=pd.read_csv("test.csv")
test=test.values.astype('float32')
test.resize(test.shape[0],28,28,1)

#standardize data
# train_labels=train_labels.ravel()#to flatten the array
train_labels=pd.get_dummies(train_labels)
train_labels.astype('int32')
print(train_labels.head())


#building model
model=Sequential()
model.add(Convolution2D(32,3,3,activation='relu',input_shape=(28,28,1)))
print(model.output_shape)
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train,train_labels.values,batch_size=64,nb_epoch=6,verbose=1)

model.save('digit_model.h5')

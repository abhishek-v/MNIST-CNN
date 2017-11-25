import pandas as pd
import matplotlib
import numpy as np
import h5py as h5py
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Convolution2D,Dense,Dropout,Flatten
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


model=load_model("digit_model.h5")
#reading output
test=pd.read_csv("test.csv")
test1=test



test=test.values.astype('float32')
test.resize(test.shape[0],28,28,1)



op=model.predict(test)

while True:
    id=int(input("Enter array index:"))
    if(id=="-1"):
        break
    else:
        a = np.array(test1.iloc[[id]])
        size = int(a.shape[1] ** (1 / 2))
        a.resize(size, size)
        plt.imshow(a)
        plt.show()
        print(np.argmax(op[id]))

# print(op.shape)



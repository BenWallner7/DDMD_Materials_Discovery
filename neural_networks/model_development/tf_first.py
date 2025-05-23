import numpy as np
import pandas as pd
import matplotlib as plot
import matplotlib.pyplot as plt


import os
import sklearn
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split

#np.set_printoptions(threshold=np.inf)

X = np.loadtxt("./equilibrated_run_rhos",dtype = 'float', comments = '#', ndmin=2) 

X = X / 1.2

y = np.loadtxt("./equilibrated_run_profile",dtype = 'float', comments = '#', ndmin=2) 


#y[::2] = y[::2] / 20

highest_temp = np.max(y)

#print(highest_temp)

#y[1::2] = y[1::2] / highest_temp
y = y / highest_temp

#print(y)

# X = np.loadtxt("./KAPPA/data/single_time_rhos",dtype = 'float', comments = '#', ndmin=2) 
# y = np.loadtxt("./KAPPA/data/single_time_profile",dtype = 'float', comments = '#', ndmin=2) 
y=y.reshape(len(X),20)

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.2, random_state=123)

X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

input_shape = X_train[1].shape


print("Number of features: ", input_shape)

output_shape=y_train.shape
print(output_shape)
#quit()
model = Sequential([InputLayer(input_shape=1), Dense(units=10, activation = "relu" ) , Dense(units=30, activation = "relu") , Dense(units=20, )])			
# model = Sequential([Flatten(input_shape=(1,)), Dense(units=10, activation="relu"), Dense(units=40, activation="relu")])			


model.compile(optimizer = Adam(learning_rate = 0.0002), loss='mae')

""""
checkpoint_name= './checkpoint'
checkpoint = ModelCheckpoint(filepath=checkpoint_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
"""
losses = model.fit(tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), validation_data=(X_val, y_val) , epochs=5000)

#print(X_train,y_train)

print(model.summary())

y_predict = model.predict(tf.convert_to_tensor(X_test))*highest_temp

print(X_test, y_predict)

model.save('./test_model')

# error = MeanSquaredError(y_test, model.predict(tf.convert_to_tensor(X_test))*highest_temp
# mse = error.numpy()

# print("The MSE of the model is: ",mse)

plt.subplot(1,2,1)

loss_df = pd.DataFrame(losses.history)
loss_df.loc[:,['loss','val_loss']].plot()
plt.show()

plt.subplot(1,2,2)

plt.plot(y_predict[1])
plt.plot(y_test[1]*highest_temp)
plt.legend(['prediction', 'actual'])

"""
plt.subplot(1,2,2)
plt.scatter(X_train, y_train, c="b", label="Train data")
plt.scatter(X_test, y_test, c="g", label="Test set")
plt.scatter(y_predict, c="r", label="Predictions")
plt.legend()
"""

plt.show()

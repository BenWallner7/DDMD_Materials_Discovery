import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split



X = np.loadtxt("/Users/benwa/tf_NN_codes/equilibrated_run_rhos",dtype = 'float', comments = '#', ndmin=2)
original_y = np.loadtxt("/Users/benwa/tf_NN_codes/equilibrated_run_profile",dtype = 'float', comments = '#', ndmin=2)

y = original_y.reshape(56, 20)


X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.2, random_state=123)

X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

input_shape = X_train.shape

num_of_tests = len(X_test)
num_of_validations = len(X_val)


print("Number of features: ", input_shape)

output_shape = y_train.shape

model = Sequential([
    Flatten(input_shape=(1,)),
    Dense(units=20, activation="relu"),
    Dense(units=80, activation = "relu"),
    
    Dense(units=20, )])


model.compile(optimizer=Adam(
    learning_rate=0.02
    ), loss='mse')

losses = model.fit(tf.convert_to_tensor(X_train),
          tf.convert_to_tensor(y_train),
	  validation_data=(X_val, y_val),
          epochs=500)


y_predict = model.predict(tf.convert_to_tensor(X_test))

output_predict = np.column_stack((X_test, y_predict))

print(output_predict.shape)
print(output_predict)
np.savetxt('./test.output', y_predict)
np.savetxt('./x.test.ouput', X_test)

error = MeanSquaredError()(y_test, y_predict)
mse = error.numpy()

print("The MSE of the model is: ", mse)


loss_df = pd.DataFrame(losses.history)
loss_df.loc[:,['loss','val_loss']].plot()
plt.show()


fig, ax = plt.subplots(3,2)

for i in range(num_of_tests):
    print(X_test[i], y_test[i],y_predict[i])
    plt.subplot(3,2,i+1)
    plt.plot(y_test[i],"x",label="input")
    plt.plot(y_predict[i],"x",label="predict")
    plt.legend()
    plt.axis()
    plt.ylim(0.5,5)
plt.show()

model.save('./good_fit')
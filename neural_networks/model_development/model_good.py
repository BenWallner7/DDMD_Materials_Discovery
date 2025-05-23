import numpy as np
import matplotlib as plot
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


# np.set_printoptions(threshold=np.inf)

X = np.loadtxt("./equilibrated_run_rhos",dtype = 'float', comments = '#', ndmin=2)
original_y = np.loadtxt("./equilibrated_run_profile",dtype = 'float', comments = '#', ndmin=2)
#X = np.loadtxt("./rhos.in", dtype='float', comments='#', ndmin=2)
#original_y = np.loadtxt("./data.in", dtype='float', comments='#', ndmin=2)
# print(y.shape)

# original_y = original_y
y = original_y.reshape(56, 20)


# print(y.shape)
# quit()

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.2, random_state=123)

X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

input_shape = X_train.shape


print("Number of features: ", input_shape)

output_shape = y_train.shape
print(output_shape)
# quit()
model = Sequential([
    Flatten(input_shape=(1,)),
    Dense(units=20, activation="relu"),
    Dense(units=80, activation = "relu"),
    
    Dense(units=20, )])
# model = Sequential([Flatten(input_shape=(1,)), Dense(units=20, activation="relu"), Dense(units=40, activation="relu")])



model.compile(optimizer=Adam(
    learning_rate=0.02
    ), loss='mse')

losses = model.fit(tf.convert_to_tensor(X),
          tf.convert_to_tensor(y),
	  validation_data=(X_val, y_val),
          epochs=500)

# print(X_train,y_train)

# print(model.summary())

y_predict = model.predict(tf.convert_to_tensor(X_test))

print(y_predict)
np.savetxt('./test.output', y_predict)
np.savetxt('./x.test.ouput', X_test)

error = MeanSquaredError()(y_test, y_predict)
mse = error.numpy()

print("The MSE of the model is: ", mse)

y_predict = model.predict(tf.convert_to_tensor(X))

print(y_predict)

loss_df = pd.DataFrame(losses.history)
loss_df.loc[:,['loss','val_loss']].plot()
plt.show()

index = 10

fig, ax = plt.subplots(3,4)

for i in range(11):
    index = i*5+3
        
    print(X[index],y[index],y_predict[index])
    plt.subplot(3,4,i+1)
    plt.plot(y[index],"x",label="input")
    plt.plot(y_predict[index],"x",label="predict")
    plt.legend()
    plt.axis()
    plt.ylim(0.5,5)
plt.show()

model.save('./good_fit')

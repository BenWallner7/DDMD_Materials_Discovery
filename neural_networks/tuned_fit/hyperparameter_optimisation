import pandas as pd
import sklearn
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

X = np.loadtxt("./equilibrated_run_rhos",dtype = 'float', comments = '#', ndmin=2)
original_y = np.loadtxt("./equilibrated_run_profile",dtype = 'float', comments = '#', ndmin=2)

y = original_y.reshape(56, 20)


X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.2, random_state=123)

X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

#print(X_train[1].shape)

input_shape = X_train[1].shape

def create_model(layers, activation, learning_rate):
	model = Sequential()
	for i, nodes in enumerate(layers):
		if i==0:
			model.add(Dense(nodes, input_dim = 1, activation = activation))
		else:
			model.add(Dense(nodes, activation = activation))
	model.add(Dense(20))
	model.compile(optimizer = 'adam', loss = 'mse')
	return model
model = KerasRegressor(build_fn = create_model, verbose = 1)

layers = [[20], [80,20], [60, 40, 20], [120, 80, 40], [160, 120, 80, 40],[172, 144, 120, 96, 72, 48], [220, 196, 172, 144, 120, 96, 72, 48] , [160, 120, 80, 40], [200, 160, 120, 80, 40]]
activations = ['sigmoid', 'relu']
lr = [float(x) for x in np.linspace(0.008, 0.04, num = 50)]
batches = [int(x) for x in np.linspace(2, 200, num = 30)]
n_times =  [int(x) for x in np.linspace(800, 1500, num = 30)]

param_grid = dict(layers = layers, activation = activations, learning_rate = lr, batch_size = batches, epochs = n_times)

ran_search = RandomizedSearchCV(estimator = model, 
                            param_distributions = param_grid,
                            n_iter = 50, 
                            scoring='neg_mean_squared_error',
                            cv = 5, 
                            verbose=0,
                            random_state=42,
                            n_jobs=-1, 
                            return_train_score=True)
ran_result = ran_search.fit(X_train, y_train)

y_predict = ran_result.predict(tf.convert_to_tensor(X_test))

error = MeanSquaredError()(y_test, y_predict)
mse = error.numpy()

print("The MSE of the model is: ", mse)


print('Score for hyperparameter tuning: ', ran_result.score)

best_mod = ran_search.best_params_

print('Best parameters from hyperparameter tuning: ', ran_result.best_params_)




best_mod.save('best_random')

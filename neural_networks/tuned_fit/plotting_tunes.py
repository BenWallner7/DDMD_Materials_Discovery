import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import tensorflow as tf


X = np.loadtxt("./equilibrated_run_rhos",dtype = 'float', comments = '#', ndmin=2)
original_y = np.loadtxt("./equilibrated_run_profile",dtype = 'float', comments = '#', ndmin=2)
y = original_y.reshape(56, 20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = tf.keras.models.load_model('./tuned_fit')

y_predict = model.predict(X_test)


# Temperature profile comparison
x_axis = np.arange(1,21)
print(y_test[int(np.where(X_test==1.16)[0])])

plt.plot(x_axis, y_test[int(np.where(X_test==1.16)[0])],"x",label='Actual temperature profile for rho = 1.16')

plt.plot(x_axis, y_predict[int(np.where(X_test==1.16)[0])],"x",label='Predicted temperature profile for rho = 1.16')


plt.xlabel('Chunk number (au)')
plt.xticks(np.arange(1,21))
#plt.yticks(np.arange(1,4.75,0.25))

plt.ylabel('Temperature (\xb0C)')

plt.legend(loc='upper right')
#plt.title('Plot that shows if the model can accurately predict \'rho\' values outside the range of values investigated.', pad =10)
plt.grid(True)

plt.axis()
#plt.savefig('out_range_rhos.png')

plt.show()

# Plot 2
x_axis = np.arange(1,21)
print(y_test[int(np.where(X_test==1.16)[0])])

plt.plot(x_axis, y_test[int(np.where(X_test==0.32)[0])],"x",label='Actual temperature profile for rho = 0.32')

plt.plot(x_axis, y_predict[int(np.where(X_test==0.32)[0])],"x",label='Predicted temperature profile for rho = 0.32')


plt.xlabel('Chunk number (au)')
plt.xticks(np.arange(1,21))
#plt.yticks(np.arange(1,4.75,0.25))

plt.ylabel('Temperature (\xb0C)')

plt.legend(loc='upper right')
#plt.title('Plot that shows if the model can accurately predict \'rho\' values outside the range of values investigated.', pad =10)
plt.grid(True)

plt.axis()
#plt.savefig('out_range_rhos.png')

plt.show()


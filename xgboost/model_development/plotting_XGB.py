import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
#os.environ['PATH'] += os.pathsep + "C:/Users/benwa/miniconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/"

X = np.loadtxt("./equilibrated_run_rhos",dtype = 'float', comments = '#', ndmin=2)
original_y = np.loadtxt("./equilibrated_run_profile",dtype = 'float', comments = '#', ndmin=2)
y = original_y.reshape(56, 20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = xgb.XGBRegressor()
model.load_model('./best_random_XGB.json')

y_predict = model.predict(X_test)

"""

index = 1

print(X[index],y[index],y_predict[index])

print(X[index+1],y[index+1],y_predict[index+1])

x_axis = np.arange(1,21)

plt.figure(figsize=(10,20))
print(int(X[index]))

str_below_rho = "Actual profile for rho: " + str(float(X[index])) + "nm"

plt.plot(x_axis,y[index],"x",label=str_below_rho)


random_x = X[index]+ (X[index+1]-X[index])*np.random.random()

y_predict_for_random_x = model.predict(np.array(random_x))

print(random_x,y_predict_for_random_x)

str_ran_rho = "Predicted profile for rho: " + str('%.5f' % random_x) + "nm"
plt.plot(x_axis, y_predict_for_random_x[0],"x",label=str_ran_rho)

str_above_rho = "Actual profile for rho: " + str(float(X[index+1])) + "nm"

plt.plot(x_axis, y[index+1],"x",label=str_above_rho)
plt.xlabel('Chunk number (au)')
plt.xticks(np.arange(1,21))
#plt.yticks(np.arange(1,4.75,0.25))

plt.ylabel('Temperature (\xb0C)')

plt.legend(loc='upper right')
plt.title('Plot that shows a prediction of a randomised value between two \'rho\' values included in the data set.',pad=10)
plt.grid(True)

plt.axis()
plt.savefig('in_between_rhos.png')

plt.show()

print(model)


#xgb.plot_tree(model, num_trees=150, rankdir = 'LR')
#plt.show()


for i in range(30,40):
	xgb.plot_tree(model, num_trees=i, rankdir = 'LR')
	#plt.rcParams['figure.figsize'] = [10, 10]
	plt.show()


index2 = 45

plt.figure(figsize=(10,20))
print(int(X[index2]))

str_below_rho = "Actual profile for rho: " + str(float(X[index2])) + "nm"

plt.plot(x_axis,y[index2],"x",label=str_below_rho)

str_above_rho = "Actual profile for rho: " + str(float(X[index2+10])) + "nm"

plt.plot(x_axis, y[index2+10],"x",label=str_above_rho)


y_predict_for_out_range = model.predict(np.array([1.4]))

print(1.4,y_predict_for_out_range)

str_rho_out_range = "Predicted profile for rho: " + str(1.4) + "nm"
plt.plot(x_axis, y_predict_for_out_range[0],"x",label=str_rho_out_range)

plt.xlabel('Chunk number (au)')
plt.xticks(np.arange(1,21))
#plt.yticks(np.arange(1,4.75,0.25))

plt.ylabel('Temperature (\xb0C)')

plt.legend(loc='upper right')
plt.title('Plot that shows if the model can accurately predict \'rho\' values outside the range of values investigated.', pad =10)
plt.grid(True)

plt.axis()
plt.savefig('out_range_rhos.png')

plt.show()

print(model)
"""

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
plt.grid(True)

plt.axis()
plt.title('Plots comparing the actual and predicted temperature profile for a certain value of rho in the testing dataset.', pad=10)
plt.savefig('out_range_rhos.png')


plt.show()

#Second plot

plt.plot(x_axis, y_test[int(np.where(X_test==0.62)[0])],"x",label='Actual temperature profile for rho = 0.62')

plt.plot(x_axis, y_predict[int(np.where(X_test==0.62)[0])],"x",label='Predicted temperature profile for rho = 0.62')


plt.xlabel('Chunk number (au)')
plt.xticks(np.arange(1,21))
#plt.yticks(np.arange(1,4.75,0.25))

plt.ylabel('Temperature (\xb0C)')

plt.legend(loc='upper right')
plt.title('Plots comparing the actual and predicted temperature profile for a certain value of rho in the testing dataset.', pad=10)
plt.grid(True)

plt.axis()
plt.savefig('out_range_rhos.png')

plt.show()

#Plot 3

plt.plot(x_axis, y_test[int(np.where(X_test==0.32)[0])],"x",label='Actual temperature profile for rho = 0.32')

plt.plot(x_axis, y_predict[int(np.where(X_test==0.32)[0])],"x",label='Predicted temperature profile for rho = 0.32')


plt.xlabel('Chunk number (au)')
plt.xticks(np.arange(1,21))
#plt.yticks(np.arange(1,4.75,0.25))

plt.ylabel('Temperature (\xb0C)')

plt.legend(loc='upper right')
plt.title('Plots comparing the actual and predicted temperature profile for a certain value of rho in the testing dataset.', pad=10)
plt.grid(True)

plt.axis()
plt.savefig('out_range_rhos.png')

plt.show()

#Training comparison

y_predict_train = model.predict(X_train)

plt.plot(x_axis, y_train[int(np.where(X_train==0.10)[0])],"x", label='Actual temperature profile for rho = 0.10')

plt.plot(x_axis, y_predict_train[int(np.where(X_train==0.10)[0])],"x", label='Predicted temperature profile for rho = 0.10')


plt.xlabel('Chunk number (au)')
plt.xticks(np.arange(1,21))
#plt.yticks(np.arange(1,4.75,0.25))

plt.ylabel('Temperature (\xb0C)')

plt.legend(loc='upper right')
plt.title('Plots comparing the actual and predicted temperature profile for a certain value of rho in the training dataset.', pad=10)
plt.grid(True)

plt.axis()
plt.savefig('out_range_rhos.png')

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from pandas import read_csv
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


X = np.loadtxt("./equilibrated_run_rhos",dtype = 'float', comments = '#', ndmin=2)
original_y = np.loadtxt("./equilibrated_run_profile",dtype = 'float', comments = '#', ndmin=2)
y = original_y.reshape(56, 20)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


n_estimators = [int(x) for x in np.linspace(start = 500 , stop = 900, num = 75)]
max_depth = [int(x) for x in np.linspace(4, 15, num = 13)] 
max_depth.append(None)
eta = [float(x) for x in np.linspace(0.011, 0.015, num = 50)]
subsample = [float(x) for x in np.linspace(0.20, 0.45, num = 30)]
colsample_bytree = [float(x) for x in np.linspace(0.4, 0.7, num = 30)]

r_grid = {'n_estimators': n_estimators,

               'eta': eta,

               'max_depth': max_depth,

               'subsample': subsample,

               'colsample_bytree': colsample_bytree}
#print(r_grid)

rgb = XGBRegressor(random_state=1)

rgb_random = RandomizedSearchCV(estimator=rgb, 
	     param_distributions=r_grid,
	     n_iter = 30, 
	     scoring='neg_mean_squared_error',
	     cv = 5, 
 	     refit=True,
	     verbose=0, 
	     random_state=42, 
             n_jobs=-1, 
             return_train_score=True)

rgb_random.fit(X_train, y_train);
best_rand_p = rgb_random.best_params_
print('The best parameters for the model are: ', best_rand_p)
best_random = XGBRegressor(n_estimators= best_rand_p['n_estimators'], 
			   max_depth = best_rand_p['max_depth'],
			   eta=best_rand_p['eta'], 
			   subsample= best_rand_p['subsample'], 
	 		   colsample_bytree=best_rand_p['colsample_bytree'])

best_random.fit(X_train, y_train)
print('Score for hyperparameter tuning: ', best_random.score(X_test , y_test))

#best_random.save_model('best_random')

#model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.01, subsample= 0.7, colsample_bytree=0.8)

model = best_random

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

mean_train_mse = np.absolute(rgb_random.cv_results_['mean_train_score'])

mean_test_mse = np.absolute(rgb_random.cv_results_['mean_test_score'])

plt.plot(mean_train_mse, mean_test_mse, 'x')
plt.plot(mean_train_mse[rgb_random.best_index_], mean_test_mse[rgb_random.best_index_], 'x', color='red', label = 'Best hyperparameters.')
plt.legend()
plt.xlabel("Averaged Mean Squared Error based on Training data (au)", labelpad=10)
plt.ylabel("Averaged Mean Squared Error based on Testing data (au)", labelpad=10)
plt.title('Plot showing how the Testing and Training errors vary for each set of parameters \n tested using the RandomizedSearchCV.', pad=10)
plt.xlim(0.,0.0008)
plt.ylim(0.0009,0.0016)
plt.grid(True)
plt.show()
plt.savefig('MSE_XGB')

print('best score: ', rgb_random.best_score_)

print('Index of best parameters: ', rgb_random.best_index_)



#print("Results of cross validation 'test score average' : ",rgb_random.cv_results_['mean_train_score'])

#print('The final mean squared error calculated is: ', scores["neg_mean_squared_error"].tail(1))
model.fit(X_train, y_train)

scores = np.absolute(scores)
plt.plot(scores)
print('Mean MSE: ' ,scores.mean(),'+/-', scores.std()) 

model.save_model("best_random_XGB.json")

y_predict = model.predict(X_test)


y_predict_test = model.predict(np.array([[0.1312]]))




print(y_predict_test)

#accuracy = accuracy_score(y_test, y_predict)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))


fig, ax = plt.subplots(4,3)
print(len(X_test))
print(X_test)
for i in range(len(X_test)):
    #print(X_test[i], y_test[i],y_predict[i])
    plt.subplot(4,3,i+1)
    plt.plot(y_test[i],"x", fillstyle='none', markersize=5,label="input", color='black')
    plt.plot(y_predict[i],"x", fillstyle='none', markersize=5,label="predict",color='red')
    plt.legend()
    plt.axis()
    plt.ylim(0.5,5)
    plt.xlim(0,20,1)
plt.show()


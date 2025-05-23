import numpy as np

X = np.loadtxt("./single_time_rhos",dtype = 'float', comments = '#', ndmin=2) 



y = np.loadtxt("./single_time_profile",dtype = 'float', comments = '#', ndmin=2)


X1 = np.loadtxt("./equilibrated_run_rhos",dtype = 'float', comments = '#', ndmin=2) 



y1 = np.loadtxt("./equilibrated_run_profile",dtype = 'float', comments = '#', ndmin=2)
y1 - y1.flatten()
y1=y1.reshape(56,20)

print(X.shape)
print(y.shape)

print(X1.shape)
print(y1.shape)
#print(len(y1))
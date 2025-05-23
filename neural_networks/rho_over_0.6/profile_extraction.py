import glob
import numpy as np
import os


def load_profile(file, initial_offset=3, chunk_size=20,):
    index = 0
    result = {}
    i=1
    chunk_ave=[]
    chunk_val=[]
    while True:
        try:
            info = np.loadtxt(file,
                skiprows=initial_offset + chunk_size * index+(index),
                max_rows=1,
            )
            data = np.loadtxt(file, usecols=(0,3),
                skiprows=initial_offset + chunk_size * index+(index +1),
                max_rows=chunk_size,
            )
            
            result[int(info[0])] = data
            index +=1
        except Exception as e:
            # print(e)
            break
    while i<21:
        for key, value in result.items():
            chunk_val=np.append(chunk_val,value[i-1,1])
        ave_val = np.average(chunk_val)
        chunk_ave= np.append(chunk_ave,[int(i),ave_val])
        chunk_val=[]
        i+=1
            
        
    return chunk_ave.reshape(chunk_size,2)
    file.close()    

rho_single_run_output = glob.iglob('/home/zcqsbaj/mp_tests/rho_over_0.6/rho*_sim/profile_*.mp',recursive=True)

input_x=[]
target_y=[]
for item in rho_single_run_output:
	rho = float(item[-7:-3])
	profile = load_profile(item)
	input_x = np.append(input_x,rho)
	target_y = np.append(target_y, profile)
print(input_x,target_y)

###Training machine learning model
import matplotlib as plot
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError

model = Sequential([Dense(units=5, activation="relu"),
		    Dense(units=1, activation="relu")])			

x = input_x 
y = target_y
model.compile(loss=MeanSquaredError)
model.fit(input_x,target_y)
print(model.summary())

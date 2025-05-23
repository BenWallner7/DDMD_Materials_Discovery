import glob
import numpy as np
rho_equilibrium_output = glob.iglob('/Users/benwa/tf_codes/data/profile_data/ave_temp_*', recursive=True)

input_x=np.array([])
target_y=np.array([])
for item in rho_equilibrium_output:
	rho = float(item[-4:])
	profile = np.loadtxt(item)
	
	input_x = np.append(input_x,rho)
	target_y = np.append(target_y, profile)
input_x = input_x.reshape(len(input_x))
target_y = target_y.reshape(len(input_x),40)
np.savetxt('equilibrated_run_rhos', input_x, fmt='%f')
np.savetxt('equilibrated_run_profile', target_y, fmt='%f')

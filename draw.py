# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
save_path = 'result/0/'

start_T = 2
total_T = 3000
last_T = 500

stepsize = 9
window_length = 50
delay_record = np.load(save_path+'GD_stepsize_'+str(stepsize)+'.npy')[start_T:total_T]
plt.plot(delay_record,label = 'GD',marker='v',markersize=10,markevery = 200)
print('GD: \t',np.average(delay_record[-last_T:]))

delay_record = np.load(save_path+'VM_stepsize_'+str(stepsize)+'.npy')[start_T:total_T]
plt.plot(delay_record[:],linestyle = '-',label = 'Momentum',marker='x',markersize=10,markevery = 200)
print('VM: \t',np.average(delay_record[-last_T:]))

delay_record = np.load(save_path+'MVS_stepsize_'+str(stepsize)+'_window_'+str(window_length)+'.npy')[start_T:total_T]
plt.plot(delay_record[:],linestyle = '-',label = 'MVS',marker='d',markersize=10,markevery = 200)
print('MVS: \t',np.average(delay_record[-last_T:]))
plt.grid()
plt.xlabel(r'iteration')
plt.ylabel('delay/seconds')
plt.legend()

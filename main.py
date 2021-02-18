# -*- coding: utf-8 -*-
import numpy as np
import allmodule

K = 20
OUT_EPOCH = 3000

env = allmodule.ENV(K, 500, 500)
agent = allmodule.COMMUNICATION()
N_0 = env.dBm_to_P(-174)
B_max = 1*1e7

P_max = env.dBm_to_P(10)

P_max_list = np.random.uniform(low=P_max, high=P_max, size=K)
D = 1e6*np.log(2)
C = 1e6


env_path = 'result/'
g_list = np.loadtxt(env_path+'g_list.csv', delimiter=',')
f_list = np.loadtxt(env_path+'f_list.csv', delimiter=',')

in_init = 0
origin_path = env_path+str(in_init)+'/'

x_origin = np.load(origin_path+'x_origin.npy')
y_origin = np.loadtxt(origin_path+'y_origin.csv', delimiter=',')

stepsize = 9

alpha_p = 1*(10**-stepsize)
gamma_b = [1/((i+1)**0.5) for i in range(OUT_EPOCH)]
gamma_p = [1/((i+1)**0.5) for i in range(OUT_EPOCH)]
step_coeff = [gamma_b, gamma_p]

window_length = 50

# MVS
save_file = origin_path+'MVS_stepsize_'+str(stepsize)+'_window_'+str(window_length)+'.npy'
print(save_file)

x_record, delay_record = allmodule.hybrid_momentum_window(
                                  agent, x_origin, y_origin, OUT_EPOCH,
                                  alpha_p, alpha_p, rho=1,
                                  P_max_list=P_max_list, B_max=B_max,
                                  g_list=g_list, f_list=f_list, D=D, C=C,
                                  N_0=N_0, step_coeff=step_coeff,
                                  window_length=window_length)
np.save(save_file, delay_record)

# Momentum
save_file = origin_path+'VM_stepsize_'+str(stepsize)+'.npy'
print(save_file)

x_record, delay_record = allmodule.hybrid_momentum_vanilla(
                                  agent, x_origin, y_origin, OUT_EPOCH,
                                  alpha_p, alpha_p, rho=1,
                                  P_max_list=P_max_list, B_max=B_max,
                                  g_list=g_list, f_list=f_list, D=D, C=C,
                                  N_0=N_0, step_coeff=step_coeff)
np.save(save_file, delay_record)

# GD
save_file = origin_path+'GD_stepsize_'+str(stepsize)+'.npy'
print(save_file)

x_record, delay_record = allmodule.hybrid(
                                  agent, x_origin, y_origin, OUT_EPOCH,
                                  alpha_p, alpha_p, rho=1,
                                  P_max_list=P_max_list, B_max=B_max,
                                  g_list=g_list, f_list=f_list, D=D, C=C,
                                  N_0=N_0, step_coeff=step_coeff)
np.save(save_file, delay_record)

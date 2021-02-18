# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


def SINR(p_list,g_list,b_list,N,i):
    '''
    

    Parameters
    ----------
    p_list : TYPE
        DESCRIPTION.
    g_list : p_1 = g*p_0.
    b_list : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.

    Returns
    -------
    beta : TYPE
        DESCRIPTION.

    '''
    s = np.array(p_list)*np.array(g_list)
    
    num = s[i]
    den = b_list[i]*N + np.sum(s) - num
    beta = num/den
    if beta>0:
        pass
    else:
        print(beta,p_list,b_list)
    assert beta>0 
    return beta


class COMMUNICATION():
    def __init__(self):
        pass
    def delay(self,b_list,p_list,g_list,N_0,f_list,D = 28.1*1000*np.log(2),C = 10**7):
        '''
        Calculate the delay of the K clients.

        Parameters
        ----------
        b_list : bandwidth.
        p_list : power.
        g_list : channel gain.
        N_0 : noise density.
        f_list : CPU frequency.
        D : The default is 28.1*1000*np.log(2).
        C : The default is 10**7.

        Returns
        -------
        delay : a list, delay for each user.

        '''
        K = len(b_list)
        delay = []
        for i in range(K):
            beta = SINR(p_list,g_list,b_list,N_0,i)
            
            d_i = D/(b_list[i]*np.log(1+beta)) + C/f_list[i]
            delay.append(d_i)
        return delay
    def gradient_i(self,b_list,p_list,g_list,N_0,i,D = 28.1*1000*np.log(2)):
        '''
        Calculate the gradient of client i.

        Parameters
        ----------
        b_list : bandwidth.
        p_list : power.
        g_list : channel gain.
        N_0 : noise density.
        i : client index.
        D : Communication load. The default is 28.1*1000*np.log(2).

        Returns
        -------
        gradient : a 2*K-dimension vector, gradient for b and p.

        '''
        K = len(b_list)
        
        
        beta = SINR(p_list,g_list,b_list,N_0,i)
        alpha = D/(b_list[i] * (np.log(1+beta))**2)
        d_b_list = []
        for j in range(K):
            if j==i:
                d_b = -D/(b_list[i]**2 * np.log(1+beta)) + alpha*(beta**2)/(1+beta)*N_0/(g_list[i]*p_list[i])
            else:
                d_b = 0
            d_b_list.append(d_b)
        
        d_p_list = []
        for j in range(K):
            if j==i:
                d_p = -alpha * beta/(1+beta) / p_list[i]
            else:
                d_p = alpha * (beta**2)/(1+beta) * g_list[j]/(g_list[i]*p_list[i])
            d_p_list.append(d_p)
        gradient = np.array([d_b_list,d_p_list])
        return gradient
    
    def gradient(self,b_list,p_list,g_list,N_0,y,D = 100):
        K= len(b_list)
        
        g = np.zeros((2,K))
        for i in range(K):
            g_i = self.gradient_i(b_list,p_list,g_list,N_0,i,D)
            g+= y[i]*g_i
        return g
    def gradient_y(self,delay):
        return delay
    def projection(self,b_list,p_list,B_max,P_max_list,epsilon_p = 1e-15,epsilon_b = 100):
        '''
        

        Parameters
        ----------
        b_list : TYPE
            DESCRIPTION.
        p_list : TYPE
            DESCRIPTION.
        B_max : TYPE
            DESCRIPTION.
        P_max_list : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        K = len(b_list)
        P_p = [min(p_list[i],P_max_list[i]) for i in range(K)]
        P_p = [max(epsilon_p,i) for i in P_p]
        
        
        # assert np.all(np.array(b_list)>0)
        
        P_b = self._projection_x(b_list, B_max, epsilon_b)
        return np.array([P_b,P_p])
    
        b_list = [max(epsilon_b,i) for i in b_list]
        b_list = [min(i,B_max) for i in b_list]
        
        b_list = np.array(b_list)
        total_b = np.sum(b_list)
        if total_b <= B_max:
            delta = (B_max - total_b)/K
            P_b = b_list + delta
        else:
            delta = (total_b - B_max)/K
            P_b = b_list - delta
        P_b = [max(epsilon_b,i) for i in P_b]
        return np.array([P_b,P_p])
    
    def _projection_x(self,x,sum_max,x_min = 0):
        K = len(x)
        Q = 2*matrix(np.identity(K))
        q = matrix(-2*np.array(x))
        
        G = np.zeros((K+1,K))
        G[:K,:] = -np.identity(K)
        G[-1,:] = 1
        G = matrix(G)
        
        h = matrix([0.0]*K + [float(sum_max)])
        
        sol=solvers.qp(Q, q, G, h, options = {"show_progress":False})
         ##sol['status']: 'optimal':找到了最优解; 'unknown': 没有找到，可能是达到了最大迭代次数
        x = np.array(sol['x'])
        
        x = [max(x_min,i[0]) for i in x]
        #print(np.sum(x))
        return x
        
    
    def projection_y(self,y,sum_max = 1,y_min = 0):
        
        return self._projection_x(y, sum_max,y_min)
        
        
        
        y = [max(y_min,i) for i in y]
        y = [min(i,sum_max) for i in y]
        total_y = np.sum(y)
        delta = sum_max - total_y
        y = np.array(y)
        y = y+delta
        y = [max(y_min,i) for i in y]
        
        return y
        pass
class ENV():
    
    def __init__(self,K,width = 500,length=500):
        location = np.zeros(shape=(K,2))
        location[:,0] = np.random.uniform(-width/2,width/2,size=K)
        location[:,1] = np.random.uniform(-length/2,length/2,size=K)
        self.location = location
        self.distance = [self.get_distance(location[i,:],(0,0)) for i in range(K)]
        
    def get_channelloss(self,scale = 8,db = False):
        g = [self._get_pathloss(d)+self._get_shadowfading(scale) for d in self.distance]
        if not db:
            g = [10**(-i/10) for i in g]
        else:
            g = [-i for i in g]
        return g
            
    def P_to_dBm(self,P):
        return 10*np.log10(P/0.001)
    def dBm_to_P(self,x):
        return 10**(-3+x/10)
    def get_distance(self,l1,l2):
        x1,y1 = l1
        x2,y2 = l2
        return ((x1-x2)**2+(y1-y2)**2)**0.5
    
    def _get_shadowfading(self,scale=8):
        return np.random.rayleigh(scale=scale)
        
    def _get_pathloss(self,d):
        '''
        128.1+37.6*log10(d/1000)

        Parameters
        ----------
        d : m.

        Returns
        -------
        dB.

        '''
        return 128.7+37.6*np.log10(d/1000)
    
    def draw_location(self,width = 500, length = 500):
        
        plt.scatter(self.location[:,0],self.location[:,1],label='client',s=200)
        plt.scatter([0],[0],marker='^',color='black',label='BS',s=200)
        plt.xlabel('/meters',fontsize=20)
        plt.ylabel('/meters',fontsize=20)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim(-width/2,width/2)
        plt.ylim(-length/2,length/2)
        plt.grid()




def hybrid(agent,x_init,y_init,epoch,alpha_b,alpha_p,rho=1,epsilon_p = 1e-5,epsilon_b = 1000,
            B_max = 1e6,P_max_list = None,N_0 = 1e-10,f_list = None,D = np.log(2)*1e4,C = 1e4,
            g_list = None,step_coeff = None):
    x_record = [x_init]
    
    delay = agent.delay(x_init[0], x_init[1], g_list, N_0, f_list,D = D,C =C )
    delay_record = [max(delay)]
    
    alpha_bp = np.array([[alpha_b],[alpha_p]])
    
    x_old = x_init
    y_old = y_init
    for i in range(epoch):
        
        gradient = agent.gradient(x_old[0],x_old[1],g_list,N_0,y_old, D = D)
        
        
        gamma_b = step_coeff[0][i]
        gamma_p = step_coeff[1][i]
        
        gamma = np.array([[gamma_b],[gamma_p]])
        x_new = x_old - alpha_bp *gamma* gradient
        x_new = agent.projection(x_new[0], x_new[1], B_max, P_max_list,epsilon_p = epsilon_p,epsilon_b = epsilon_b)
        
        
        delay = agent.delay(x_new[0], x_new[1], g_list, N_0, f_list,D = D,C =C )
        
        y_new = (delay+y_old/rho)/(1/rho+(i+1)**(-0.25))
        y_new = np.array(agent.projection_y(y_new))
        
        delay_record.append(max(delay))
        x_record.append(x_new)
        
        x_old = x_new
        y_old = y_new
    return x_record,delay_record

def hybrid_momentum_vanilla(agent,x_init,y_init,epoch,alpha_b,alpha_p,rho=1,epsilon_p = 1e-5,epsilon_b = 1000,
            B_max = 1e6,P_max_list = None,N_0 = 1e-10,f_list = None,D = np.log(2)*1e4,C = 1e4,
            g_list = None,
            step_coeff = None,rho_gradient = 0.9):

    x_record = [x_init]
    
    delay = agent.delay(x_init[0], x_init[1], g_list, N_0, f_list,D = D,C =C )
    delay_record = [max(delay)]
    
    alpha_bp = np.array([[alpha_b],[alpha_p]])
    
    x_old = x_init
    y_old = y_init
    
    his_grad = np.zeros_like(x_old)
    for ite in range(epoch):        
        gradient_curr = agent.gradient(x_old[0],x_old[1],g_list,N_0,y_old, D = D)

        
        gamma_b = step_coeff[0][ite]
        gamma_p = step_coeff[1][ite]
        
        
        
        gamma = np.array([[gamma_b],[gamma_p]])
        x_new = x_old - alpha_bp *gamma* (gradient_curr+his_grad)
        x_new = agent.projection(x_new[0], x_new[1], B_max, P_max_list,epsilon_p = epsilon_p,epsilon_b = epsilon_b)
        
        
        delay = agent.delay(x_new[0], x_new[1], g_list, N_0, f_list,D = D,C =C )
        
        y_new = (delay+y_old/rho)/(1/rho+(ite+1)**(-0.25))
        y_new = np.array(agent.projection_y(y_new))
        
        delay_record.append(max(delay))
        x_record.append(x_new)
        
        x_old = x_new
        y_old = y_new
        
        his_grad = rho_gradient*(his_grad + gradient_curr)
    return x_record,delay_record




def hybrid_momentum(agent,x_init,y_init,epoch,alpha_b,alpha_p,rho=1,epsilon_p = 1e-5,epsilon_b = 1000,
            B_max = 1e6,P_max_list = None,N_0 = 1e-10,f_list = None,D = np.log(2)*1e4,C = 1e4,
            g_list = None,
            step_coeff = None,rho_gradient = 0.9,epsilon_gradient = 0.9):
    
    
    def cal_sim(x,y):
        dis = np.linalg.norm(x-y,ord=2)
        
        return np.exp(-dis)
    def gi_to_g(g,y):
        #g:[(K,2),(K,2),...,] T
        #y: (K,)
        rg = np.zeros_like(g[0])
        for i in range(len(y)):
            rg += y[i]*g[i]
        return rg
    
    def get_his_g(y_now):
        T = len(grad_record)
        h = np.zeros_like(x_init)
        dis = [cal_sim(yi,y_now) for yi in y_record]
        if len(dis)>3:
            pass
            #print(ite)
            #print([round(i,3) for i in y_now],round(dis[-1],2),round(dis[-2],2))
        temp_g = [gi_to_g(his_gi,y_now) for his_gi in grad_record]
        for t in range(T):
            if dis[t]>epsilon_gradient:
                h+= (rho_gradient**(T-t))*dis[t]*temp_g[t]
        #print(round(np.linalg.norm(h),2))
        return h

    x_record = [x_init]
    
    delay = agent.delay(x_init[0], x_init[1], g_list, N_0, f_list,D = D,C =C )
    delay_record = [max(delay)]
    
    alpha_bp = np.array([[alpha_b],[alpha_p]])
    
    x_old = x_init
    y_old = y_init
    
    K = len(x_old[0])
    grad_record = []
    y_record = []
    for ite in range(epoch):
        
        g_i = [agent.gradient_i(x_old[0], x_old[1], g_list, N_0, yi,D = D) for yi in range(K)]
        
        gradient_curr = gi_to_g(g_i,y_old)
        
        
        g_t = agent.gradient(x_old[0],x_old[1],g_list,N_0,y_old, D = D)
        
        error = np.linalg.norm(g_t - gradient_curr)
        #print('grad,should = 0',error)
        assert error<1e-3
        
        np.linalg.norm(y_old,ord = 2)
        
        if grad_record:
            his_grad = get_his_g(y_old)
            
        else:
            his_grad = np.zeros_like(x_old)
            
                
            
        grad_record.append(g_i)
        y_record.append(y_old)
        #print(gradient)
        #gamma_b = (0.99)**int(i/10)
        #gamma_p = (0.99)**int(i/10)
        
        gamma_b = step_coeff[0][ite]
        gamma_p = step_coeff[1][ite]
        
        gamma = np.array([[gamma_b],[gamma_p]])
        x_new = x_old - alpha_bp *gamma* (gradient_curr+his_grad)
        x_new = agent.projection(x_new[0], x_new[1], B_max, P_max_list,epsilon_p = epsilon_p,epsilon_b = epsilon_b)
        
        
        delay = agent.delay(x_new[0], x_new[1], g_list, N_0, f_list,D = D,C =C )
        
        y_new = (delay+y_old/rho)/(1/rho+(ite+1)**(-0.25))
        y_new = np.array(agent.projection_y(y_new))
        
        delay_record.append(max(delay))
        x_record.append(x_new)
        
        x_old = x_new
        y_old = y_new
    return x_record,delay_record


def hybrid_momentum_window(agent,x_init,y_init,epoch,alpha_b,alpha_p,rho=1,epsilon_p = 1e-5,epsilon_b = 1000,
            B_max = 1e6,P_max_list = None,N_0 = 1e-10,f_list = None,D = np.log(2)*1e4,C = 1e4,
            g_list = None,
            step_coeff = None,rho_gradient = 0.9,epsilon_gradient = 0.9,window_length = 100):
    
    
    def cal_sim(x,y):
        dis = np.linalg.norm(x-y,ord=2)
        
        return np.exp(-dis)
    def gi_to_g(g,y):
        #g:[(K,2),(K,2),...,] T
        #y: (K,)
        rg = np.zeros_like(g[0])
        for i in range(K):
            rg += y[i]*g[i]
        return rg
    
    def get_his_g():
        T = window_length
        h = np.zeros_like(x_init)
        dis = [cal_sim(yi,y_old) for yi in y_record]
        if len(dis)>3:
            pass
            #print(ite)
            #print([round(i,3) for i in y_old],round(dis[-1],2),round(dis[-2],2))
        temp_g = [gi_to_g(his_gi,y_old) for his_gi in grad_record]
        for t in range(T):
            if dis[t]>epsilon_gradient:
                h+= (rho_gradient**(T-t))*dis[t]*temp_g[t]
        #print(round(np.linalg.norm(h),2))
        return h

    x_record = [x_init]
    
    delay = agent.delay(x_init[0], x_init[1], g_list, N_0, f_list,D = D,C =C )
    delay_record = [max(delay)]
    
    alpha_bp = np.array([[alpha_b],[alpha_p]])
    
    x_old = x_init
    y_old = y_init
    
    K = len(x_old[0])
    grad_record = [[np.zeros_like(x_old) for yi in range(K)]]*window_length
    y_record = [np.zeros_like(y_old)-1]*window_length
    for ite in range(epoch):
        
        g_i = [agent.gradient_i(x_old[0], x_old[1], g_list, N_0, yi,D = D) for yi in range(K)]
        
        gradient_curr = gi_to_g(g_i,y_old)
        
        
        g_t = agent.gradient(x_old[0],x_old[1],g_list,N_0,y_old, D = D)
        error = np.linalg.norm(g_t - gradient_curr)        
        assert error<1e-3
        
        
        his_grad = get_his_g()
        
        
        for w_i in range(window_length-1):
            grad_record[w_i] = grad_record[w_i+1]
            y_record[w_i] = y_record[w_i+1]
        
        grad_record[-1] = g_i
        y_record[-1] = y_old

        gamma_b = step_coeff[0][ite]
        gamma_p = step_coeff[1][ite]
        
        gamma = np.array([[gamma_b],[gamma_p]])
        x_new = x_old - alpha_bp *gamma* (gradient_curr+his_grad)
        x_new = agent.projection(x_new[0], x_new[1], B_max, P_max_list,epsilon_p = epsilon_p,epsilon_b = epsilon_b)
        
        
        delay = agent.delay(x_new[0], x_new[1], g_list, N_0, f_list,D = D,C =C )
        
        y_new = (delay+y_old/rho)/(1/rho+(ite+1)**(-0.25))
        y_new = np.array(agent.projection_y(y_new))
        
        delay_record.append(max(delay))
        x_record.append(x_new)
        
        x_old = x_new
        y_old = y_new
    return x_record,delay_record




if __name__=='__main__':
    
    pass
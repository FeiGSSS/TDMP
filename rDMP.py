# -*- encoding: utf-8 -*-
'''
@File    :   rDMP.py
@Time    :   2022/03/31 10:38:23
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import EoN
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def _my_odeint_(dfunc, V0, times, args=()):
    r = integrate.ode(lambda t, X: dfunc(X, t, *args))
    r.set_integrator('vode', method='adams')
    r.set_initial_value(V0,times[0])
    V=[V0]
    for time in times[1:]:
        V.append(r.integrate(time))
    V = np.array(V)
    return V


def _dSIS_DMP_based_(V, t, G, nodelist, index_of_node, trans_rate_fxn, rec_rate_fxn):
    #print(t)
    N = G.order()
    Ii = V[0:N]
    Iji = V[N:].reshape(N,N)
    
    dIi = np.zeros(N)
    dIji = np.zeros((N,N))
    
    for u in nodelist:
        i = index_of_node[u]
        dIi[i] += -rec_rate_fxn(u)*Ii[i] 
        for v in G.neighbors(u):
            j = index_of_node[v]
            
            dIi[i] += trans_rate_fxn(u,v) * (1-Ii[i]) * Iji[j, i]
            
            dIji[j, i] += - rec_rate_fxn(v)*Iji[j,i]
            pair_message = 0
            for w in G.neighbors(v):
                if w == u: #skip these
                    continue
                #so w != v. 
                k= index_of_node[w]
                # dIji[j, i] += trans_rate_fxn(v, w) * (1-Ii[j]) * Iji[k, j]
                # dIji[j, i] += trans_rate_fxn(v, w) * (1-Iji[j,i]) * Iji[k, j]
                pair_message += Iji[k, j]
            # dIji[j, i] += trans_rate_fxn(v, w) * (1-Iji[j,i]) * pair_message
            dIji[j, i] += trans_rate_fxn(v, w) * (1-Ii[j]) * pair_message
                
    dIi.shape = (N, 1)
    dIji.shape = (N*N, 1)
    dV = np.concatenate((dIi, dIji), axis=0).T[0]
    return dV


def SIS_DMP_based(G, tau, gamma, rho = None, nodelist = None, Y0=None, tmin = 0, tmax = 100,  
                    tcount = 1001, transmission_weight=None, recovery_weight=None, return_full_data = False):
    N = G.order()
        
    if Y0 is None and rho is None:
        rho = 1./N
    if Y0 is not None and rho is not None:
        raise EoN.EoNError("either Y0 or rho must be defined")
    if Y0 is not None and  nodelist is None:
        raise EoN.EoNError("cannot define Y0 without nodelist")
        
    if  nodelist is None: #only get here if Y0 is None
        nodelist = G.nodes()
        Y0 = np.array([rho]*N)
    if len(Y0) != N:
        raise EoN.EoNError("incompatible length for Y0")            

    trans_rate_fxn, rec_rate_fxn = EoN._get_rate_functions_(G, tau, gamma, 
                                                transmission_weight,
                                                recovery_weight)

    times = np.linspace(tmin,tmax,tcount)

    Ii = Y0.reshape(N, 1)
    Iji = np.zeros((N, N))
    Iji[np.nonzero(Y0)[0], :] = 1
    Iji.shape = (N*N, 1)
    
    V0 = np.concatenate((Ii, Iji), axis=0).T[0]
    index_of_node = {node:i for i, node in enumerate(nodelist)}

    V = _my_odeint_(_dSIS_DMP_based_, V0, times, args = (G, nodelist, index_of_node, trans_rate_fxn, rec_rate_fxn))
    Ii = V.T[0:N] # [N, times]
    I = Ii.sum(axis=0)
    Is = np.ones(N)[:,None]-Ii
    S = Is.sum(axis=0)
    return times, S, I



def SIS_DMP_based_pure_IC(G, tau, gamma, initial_infecteds, nodelist = None,
                            tmin = 0, tmax = 100, tcount = 1001,
                            transmission_weight = None, recovery_weight=None,
                            return_full_data = False):
    if nodelist is None:
        nodelist = G.nodes()
        N = len(nodelist)
        #make Y0[u] be 1 if infected 0 if not
        initial_infecteds = set(initial_infecteds) #for fast membership test
        Y0 = np.array([1 if u in initial_infecteds else 0 for u in nodelist])    

        return SIS_DMP_based(G, tau, gamma, nodelist = nodelist,
                                Y0=Y0, tmin=tmin, tmax=tmax, tcount=tcount,
                                transmission_weight=transmission_weight, 
                                recovery_weight=recovery_weight,
                                return_full_data=return_full_data)
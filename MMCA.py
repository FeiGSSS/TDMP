# -*- encoding: utf-8 -*-
'''
@File    :   MMCA.py
@Time    :   2022/03/31 17:39:53
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import numpy as np
import EoN

def _SIS_MMCA_based_(Pi, G, nodelist, index_of_node, trans_rate_fxn, rec_rate_fxn):
    N = G.order()
    Pi_new = np.zeros(N)
    for u in nodelist:
        i = index_of_node[u]
        Pi_new[i] += (1-rec_rate_fxn(u))*Pi[i]
        Qi = 1
        for v in G.neighbors(u):
            j = index_of_node[v]
            Qi *= (1 - trans_rate_fxn(u, v) * Pi[j])
        Pi_new[i] += (1 - Qi)*(1 - Pi[i])
    return Pi_new 

def SIS_MMCA_based(G, tau, gamma, nodelist, p0, tmax, transmission_weight, recovery_weight, **kargs):
    N = G.order()         
    trans_rate_fxn, rec_rate_fxn = EoN._get_rate_functions_(G, tau, gamma, 
                                                            transmission_weight,
                                                            recovery_weight)
    index_of_node = {node:i for i, node in enumerate(nodelist)}

    Pi = p0.reshape(N)
    Ii = [Pi.reshape(N, 1)]
    for _ in range(1, tmax):
        Pi = _SIS_MMCA_based_(Pi, G, nodelist, index_of_node, trans_rate_fxn, rec_rate_fxn)
        Ii.append(Pi.reshape(N, 1))
    Ii = np.concatenate(Ii, axis=1)
    I = Ii.sum(axis=0)
    Is = np.ones(N)[:,None]-Ii
    S = Is.sum(axis=0)
    times = np.arange(0, tmax)
    return times, S, I

def SIS_MMCA_based_pure_IC(G, tau, gamma, initial_infecteds, nodelist = None,
                            tmin = 0, tmax = 100, tcount = 1001,
                            transmission_weight = None, recovery_weight=None,
                            return_full_data = False):
    if nodelist is None:
        nodelist = G.nodes()
        N = len(nodelist)
        #make Y0[u] be 1 if infected 0 if not
        initial_infecteds = set(initial_infecteds) #for fast membership test
        p0 = np.array([1 if u in initial_infecteds else 0 for u in nodelist])    

        return SIS_MMCA_based(G, tau, gamma, nodelist = nodelist, p0=p0, tmin=tmin, tmax=tmax, tcount=tcount,
                                transmission_weight=transmission_weight, 
                                recovery_weight=recovery_weight,
                                return_full_data=return_full_data)
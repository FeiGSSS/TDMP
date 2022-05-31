import numpy as np
from numba import types
from numba.typed import Dict

from methods.utils import convert2nb_dict, ifConverge
from methods.rDMP.base import SIS_DMP_base

def SIS_rDMP(N:int, beta:float, gamma:float, node_neig_dict, tmax:int, I0, steady:bool=True):

    node_neig_dict = convert2nb_dict(node_neig_dict)
    
    # I(i)
    node_p = Dict.empty(key_type=types.int64,  value_type=types.float64)
    for i in range(N):
        if i in I0:
            node_p[i] = 1
        else:
            node_p[i] = 0
            
    # I(j->i)
    edge_p = Dict.empty(key_type=types.UniTuple(types.int64, 2), 
                        value_type=types.float64)
    for i in range(N):
        for j in node_neig_dict[i]:
            if j in I0:
                edge_p[(j, i)] = 1
            else:
                edge_p[(j ,i)] = 0              

    rho = np.zeros(tmax)
    rho[0] = sum(node_p.values())/N

    for t in range(1, tmax):
        node_p, edge_p = SIS_DMP_base(N, gamma, beta, node_p, edge_p, node_neig_dict)
        rho[t] = sum(node_p.values())/N
        # if ifConverge(rho[:t], N):
        #     rho = rho[:t]
        #     break

    if steady:
        out = np.mean(rho[-100:])
    else:
        out = rho
    return beta, out
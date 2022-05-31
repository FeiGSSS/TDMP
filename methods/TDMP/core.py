# -*- encoding: utf-8 -*-
'''
@File    :   core.py
@Time    :   2022/05/31 14:43:52
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import numpy as np
from numba import types
from numba.typed import Dict

from methods.utils import convert2nb_dict, ifConverge
from methods.TDMP.base import SIS_DMP_base

def SIS_TDMP(N, beta, gamma, node_edge_dict, node_tri_dict, 
             tmax, I0, steady:bool=True):
    node_edge_dict = convert2nb_dict(node_edge_dict)
    node_tri_dict  = convert2nb_dict(node_tri_dict)

    edge_keys = np.array(list(node_edge_dict.keys()))
    tri_keys  = np.array(list(node_tri_dict.keys()))

    # I(i)
    node_p = Dict.empty(key_type=types.int64, 
                        value_type=types.float64)
    for i in range(N):
        node_p[i] = 1 if i in I0 else 0
            
    # I(j->i)
    edge_p = Dict.empty(key_type=types.UniTuple(types.int64, 2), 
                        value_type=types.float64)
    for i in edge_keys:
        for j in node_edge_dict[i]:
            edge_p[(j, i)] = 1 if j in I0 else 0

    # II({j,k}->i)
    ii_p = Dict.empty(key_type=types.UniTuple(types.int64, 3),
                       value_type=types.float64)
    for i in tri_keys:
        for (j,k) in node_tri_dict[i]:
            (j, k) = sorted([j ,k])
            ii_p[(j,k,i)] = 1 if j in I0 and k in I0 else 0
    
    is_p = Dict.empty(key_type=types.UniTuple(types.int64, 3),
                       value_type=types.float64)
    for i in tri_keys:
        for (j,k) in node_tri_dict[i]:
            if j in I0 and k not in I0:
                is_p[(j,k,i)] = 1
                is_p[(k,j,i)] = 0
            elif j not in I0 and k in I0:
                is_p[(k,j,i)] = 1
                is_p[(j,k,i)] = 0
            else:
                is_p[(k,j,i)] = 0
                is_p[(j,k,i)] = 0                

    rho = np.zeros(tmax)
    rho[0] = sum(node_p.values())/N

    for t in range(1, tmax):
        node_p, edge_p, ii_p, is_p = SIS_DMP_base(N, gamma, beta, node_p, edge_p, ii_p, is_p, 
                                                  node_edge_dict, node_tri_dict, edge_keys, tri_keys)
        rho[t] = sum(node_p.values())/N
        # if ifConverge(rho[:t], N):
        #     rho = rho[:t]
        #     break

    if steady:
        out = np.mean(rho[-100:])
    else:
        out = rho
    return beta, out


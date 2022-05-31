from email import header


# -*- encoding: utf-8 -*-
'''
@File    :   core.py
@Time    :   2022/05/31 11:41:48
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''


import numpy as np
from numba import types
from numba.typed import Dict
from methods.utils import convert2nb_dict, ifConverge
from methods.ELE.base import SIS_ELE_base


def SIS_ELE(N, beta, gamma, node_neig_dict, tmax, I0, steady=True):
    node_neig_dict = convert2nb_dict(node_neig_dict)

    PI = Dict.empty(key_type=types.int64, value_type=types.float64)
    for i in range(N):
        PI[i] = 1 if i in I0 else 0
    
    PII = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    for i in range(N):
        for j in node_neig_dict[i]:
            i1, j1 = sorted([i,j])
            if i1 in I0 and j1 in I0:
                PII[(i1,j1)] = 1
            else:
                PII[(i1,j1)] = 0

    rho = np.zeros(tmax)
    rho[0] = sum(PI.values()) / N
    for t in range(1, tmax):
        PI, PII = SIS_ELE_base(N, gamma, beta, PI, PII, node_neig_dict)
        rho[t] = sum(PI.values()) / N
        # if ifConverge(rho[:t], N):
        #     rho = rho[:t]
        #     break
    
    if steady:
        out = np.mean(rho[-100:])
    else:
        out = rho
    return beta, out
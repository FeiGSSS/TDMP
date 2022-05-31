# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/05/31 11:06:29
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import numpy as np

from numba.typed import Dict 
from numba import types, njit

def convert2nb_dict(dictory):
    keys_shape = [len(v.shape) for v in dictory.values()]
    if max(keys_shape)==2:
        dict_nb = Dict.empty(key_type=types.int64, value_type=types.int64[:, :])
    elif max(keys_shape)==1:
        dict_nb = Dict.empty(key_type=types.int64, value_type=types.int64[:])
    else:
        raise ValueError

    for k, v in dictory.items():
        if len(v) == 0:
            continue
        else:
            dict_nb[int(k)] = np.array(v)
            
    return dict_nb

@njit()
def ifConverge(rho:np.ndarray, N:int, threshold:float=1e-3)->bool:
    if len(rho) < 100:
        return False
    else:
        std = np.std(rho[-100:]) * N
        if std < threshold:
            return True
        else:
            return False

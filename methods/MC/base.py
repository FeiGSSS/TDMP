
import random
from statistics import mean
import numpy as np
import time

from numba import njit
from numba.typed import Dict

from methods.utils import convert2nb_dict


@njit()
def iter(rho, tmax, I, N, gamma, beta, node_dict):
    for t in range(1, tmax):
        if np.sum(I) < 1e-4 or abs(np.sum(I)-N)<1:
            for t2 in range(t, tmax):
                rho[t2] = np.mean(I)
            return rho
        
        I_new = I.copy()
        for node in range(N):
            if I[node] == 1:  # infected
                if random.random() < gamma:
                    I_new[node] = 0
            else:
                for v in node_dict[node]:
                    if I[v] == 1 and random.random() < beta:
                        I_new[node] = 1
                        break
        I = I_new.copy()
        rho[t] = np.mean(I)
    return rho



def SIS_MC_once(seed, N, beta, gamma, node_dict, tmax, I0, steady):
    random.seed(seed)
    np.random.seed(seed)
    # init
    I = np.zeros(N)
    I[I0] = 1
    
    rho = np.zeros(tmax, dtype=np.float64)
    rho[0] = np.mean(I)
    rho = iter(rho, tmax, I, N, gamma, beta, node_dict)

    if not steady:
        out = rho
    else:
        if rho[-1] < 1e-3 or rho[-1] > 0.999:
            out = rho[-1]
        else:
            out = np.mean(rho[-int(tmax*0.1):])

    return out


def SIS_MC_batch(batch_size, seed, N, beta, gamma, node_neig_dict, tmax, I0, steady):
    node_neig_dict = convert2nb_dict(node_neig_dict)
    rhos = []
    for b in range(batch_size):
        bseed = seed + b + int(time.time() * 1000) % 1000
        rhos.append(SIS_MC_once(bseed, N, beta, gamma, node_neig_dict, tmax, I0, steady))
    
    return rhos
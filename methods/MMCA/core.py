import numpy as np
from methods.utils import convert2nb_dict, ifConverge
from numba import njit

@njit()
def update(N, I, beta, gamma, node_neig_dict):
    Inew = I.copy()
    for i in range(N):
        qi = 1
        for j in node_neig_dict[i]:
            qi *= 1 - beta * I[j]
        Inew[i] = (1 - I[i]) * (1 - qi) + I[i] * (1 - gamma)
    return Inew


def SIS_MMCA(N, beta, gamma, node_neig_dict, tmax, I0, steady:bool=True):
    node_neig_dict = convert2nb_dict(node_neig_dict)
    I = np.zeros(N)
    for idx in I0:
        I[idx] = 1

    rho = np.zeros(tmax)         
    rho[0] = np.mean(I)
    for t in range(1, tmax):
        I = update(N, I, beta, gamma, node_neig_dict)
        rho[t] = np.mean(I)
        # if ifConverge(rho[:t], N):
        #     rho = rho[:t]
        #     break
    
    if steady:
        out = np.mean(rho[-100:])
    else:
        out = rho
    
    return beta, out
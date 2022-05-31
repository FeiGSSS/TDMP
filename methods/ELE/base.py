
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

@njit
def update_P(PI, PII, PSI, PSS):
    for (i,j), _ in PII.items():
        PSI[(i,j)] = PI[j] - PII[(i,j)]
        PSI[(j,i)] = PI[i] - PII[(i,j)]
        PSS[(i,j)] = 1 -  PSI[(j,i)] - PSI[(i,j)] - PII[(i,j)]

@njit
def update_qi(N, PI, PSI, beta, node_neig_dict, qi):
    for i in range(N):
        if PI[i] == 1:
            qi[i] = 1
        else:
            condition = 1-PI[i]
            value = 1
            for j in node_neig_dict[i]:
                value *= (1 - beta * PSI[(i,j)] / condition)
            qi[i] = value

@njit
def update_qij(PI, PSI, beta, qi, qij):
    for (i, j) in PSI.keys():
        if 1 == PI[i]:
            qij[(i, j)] = 1

        else:
            condition = 1 - PI[i]
            qij[(i, j)] = qi[i] / (1 - beta * PSI[(i,j)] / condition)

@njit
def update_PI(N, PI, gamma, qi, PI_new):
    for i in range(N):
        PI_new[i] = (1 - PI[i]) * (1 - qi[i]) + PI[i] * (1 - gamma)

@njit
def update_PII(PSS, PSI, PII, qij, beta, gamma, PII_new):
    for (i, j) in PII.keys():
        part1 = PSS[(i, j)] * (1 - qij[(i,j)]) * (1 - qij[(j,i)])
        part2 = PSI[(i, j)] * (1 - (1 - beta) * qij[(i, j)]) * (1 - gamma)
        part3 = PSI[(j, i)] * (1 - (1 - beta) * qij[(j, i)]) * (1 - gamma)
        part4 = PII[(i, j)] * (1 - gamma)**2
        PII_new[(i, j)] = part1 + part2 + part3 + part4


def SIS_ELE_base(N, gamma, beta, PI, PII, node_neig_dict):

    PSI = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    PSS = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    update_P(PI, PII, PSI, PSS)

    qi = Dict.empty(key_type=types.int64, value_type=types.float64)
    update_qi(N, PI, PSI, beta, node_neig_dict, qi)
    

    qij = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    update_qij(PI, PSI, beta, qi, qij)

    PI_new = Dict.empty(key_type=types.int64, value_type=types.float64)
    update_PI(N, PI, gamma, qi, PI_new)

    PII_new = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    update_PII(PSS, PSI, PII, qij, beta, gamma, PII_new)

    return PI_new, PII_new
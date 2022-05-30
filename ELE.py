# -*- encoding: utf-8 -*-
'''
@File    :   ELE.py
@Time    :   2022/04/03 17:12:20
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import numpy as np
import EoN


def _SIS_ELE_based_(P, G, nodelist, index_of_node, trans_rate_fxn,
                    rec_rate_fxn):
    N = G.order()
    PI, PSI, PIS, PII = P
    PSS = np.ones((N, N)) - PSI - PIS - PII

    # Step 0, Hij
    Hij = np.zeros_like(PSI, dtype=float)
    for i in range(N):
        for j in range(N):
            if PI[i] == 1:
                Hij[i, j] = 0
            else:
                Hij[i, j] = PSI[i, j] / (1 - PI[i])

    # Step 1
    qij = np.ones((N, N))
    for u in nodelist:
        i = index_of_node[u]
        neighbors_index = [index_of_node[v] for v in G.neighbors(u)]
        neighbors_values = [
            1 - trans_rate_fxn(v, u) * Hij[i, index_of_node[v]]
            for v in G.neighbors(u)
        ]
        total_values = np.prod(neighbors_values)
        for j, value in zip(neighbors_index, neighbors_values):
            qij[i, j] = total_values / value

    # Step 3 update P**
    PSI_new = np.zeros_like(PSI, dtype=float)
    PIS_new = np.zeros_like(PIS, dtype=float)
    PII_new = np.zeros_like(PII, dtype=float)

    for u in nodelist:
        i = index_of_node[u]
        for v in G.neighbors(u):
            j = index_of_node[v]
            PSI_new[i, j] = PSS[i,j] * qij[i,j] * (1-qij[j,i]) + \
                            PSI[i,j] * (1-trans_rate_fxn(v, u)) * qij[i,j] * (1-rec_rate_fxn(v)) + \
                            PIS[i,j] * (1-(1-trans_rate_fxn(v,u))*qij[j,i]) * rec_rate_fxn(u) + \
                            PII[i,j] * rec_rate_fxn(u) * (1-rec_rate_fxn(v))

            PIS_new[i, j] = PSS[i,j] * (1-qij[i,j]) * qij[j,i] + \
                            PSI[i,j] * (1-(1-trans_rate_fxn(v, u)) * qij[i,j]) * rec_rate_fxn(v) + \
                            PIS[i,j] * (1-rec_rate_fxn(u)) * (1-trans_rate_fxn(u, v))*qij[j, i] + \
                            PII[i,j] * (1-rec_rate_fxn(u)) * rec_rate_fxn(v)
            
            PII_new[i, j] = PSS[i,j] * (1-qij[i,j]) * (1-qij[j,i]) + \
                            PSI[i,j] * (1-(1-trans_rate_fxn(v, u))*qij[i,j]) * (1-rec_rate_fxn(v)) + \
                            PIS[i,j] * (1-rec_rate_fxn(u)) * (1-(1-trans_rate_fxn(v, u))*qij[j,i]) + \
                            PII[i,j] * (1-rec_rate_fxn(u)) * (1-rec_rate_fxn(v))

    # Step 4: qi
    qi = np.ones_like(PI, dtype=float)
    for u in nodelist:
        i = index_of_node[u]
        qi[i] = np.prod([(1 - trans_rate_fxn(u, v) * Hij[i, index_of_node[v]])
                         for v in G.neighbors(u)])

    mu = np.zeros_like(PI, dtype=float)
    for u in nodelist:
        i = index_of_node[u]
        mu[i] = rec_rate_fxn(u)
    PI_new = (1 - PI) * (1 - qi) + PI * (1 - mu)

    P_new = [PI_new, PSI_new, PIS_new, PII_new]
    return P_new


def SIS_ELE_based_pure_IC(G,
                          tau,
                          gamma,
                          initial_infecteds,
                          nodelist=None,
                          tmin=0,
                          tmax=100,
                          tcount=1001,
                          transmission_weight=None,
                          recovery_weight=None,
                          return_full_data=False):
    nodelist = G.nodes()
    index_of_node = {node: i for i, node in enumerate(nodelist)}
    N = len(nodelist)
    initial_infecteds = set(initial_infecteds)  #for fast membership test
    PI = np.array([1 if u in initial_infecteds else 0 for u in nodelist])
    PSI = np.zeros((N, N), dtype=float)
    PII = np.zeros((N, N), dtype=float)
    PIS = np.zeros((N, N), dtype=float)
    for (u, v) in G.edges():
        i, j = index_of_node[u], index_of_node[v]
        if u in initial_infecteds and v in initial_infecteds:
            PII[i, j] = 1
            PII[j, i] = 1
        elif u in initial_infecteds and v not in initial_infecteds:
            PIS[i, j] = 1
            PSI[j, i] = 1
        elif u not in initial_infecteds and v in initial_infecteds:
            PSI[i, j] = 1
            PIS[j, i] = 1

    trans_rate_fxn, rec_rate_fxn = EoN._get_rate_functions_(
        G, tau, gamma, transmission_weight, recovery_weight)

    P = [PI, PSI, PIS, PII]
    Ii = [PI.reshape(N, 1)]
    for _ in range(1, tmax):
        P = _SIS_ELE_based_(P, G, nodelist, index_of_node, trans_rate_fxn,
                            rec_rate_fxn)
        Pi = P[0]
        Ii.append(Pi.reshape(N, 1))
    Ii = np.concatenate(Ii, axis=1)
    I = Ii.sum(axis=0)
    Is = np.ones(N)[:, None] - Ii
    S = Is.sum(axis=0)
    times = np.arange(0, tmax)
    return times, S, I
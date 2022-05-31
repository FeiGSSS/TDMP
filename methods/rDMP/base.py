import numpy as np
from numba.typed import Dict
from numba import njit


@njit()
def calculateMessages(N, beta, edge_p, node_neig_dict):
    meta_values = np.empty(N)
    for i in range(N):
        pair_messages = 1
        for j in node_neig_dict[i]:
            pair_messages *= (1 - beta * edge_p[(j,i)])

        meta_values[i] = pair_messages
    return meta_values

@njit()
def calculateCavityMessagesEdge(place_holder, beta, edge_p, node_neig_dict):
    cavity_messages = place_holder
    for (j, i), _ in cavity_messages.items():
        pair_messages = 1
        for k in node_neig_dict[j]:
            if k != i:
                pair_messages *= (1-beta * edge_p[(k, j)])
        cavity_messages[(j,i)] = pair_messages

    return cavity_messages
    

@njit()
def SIS_DMP_base(N, gamma, beta, node_p, edge_p, node_neig_dict):

    messages = calculateMessages(N, beta, edge_p, node_neig_dict)

    place_holder = edge_p.copy()
    cavity_messages_edge = calculateCavityMessagesEdge(place_holder, beta, edge_p, node_neig_dict)

    new_node_p = node_p.copy()
    for i, v in node_p.items():
        new_node_p[i]  = node_p[i] * (1 - gamma) + \
                         (1 - node_p[i]) * (1 - messages[i])
    
    new_edge_p = edge_p.copy()
    for (j, i), v in new_edge_p.items():
        new_edge_p[(j,i)] = edge_p[(j,i)] * (1 - gamma) + \
                      (1 - edge_p[(j,i)]) * (1 - cavity_messages_edge[(j, i)])
    
    return new_node_p, new_edge_p
import numpy as np
from numba.typed import Dict
from numba import njit


@njit()
def calculateMessages(N, node_edge_dict, node_tri_dict, edge_p, ii_p, is_p, beta, edge_keys, tri_keys):
    meta_values = np.empty(N)
    for i in range(N):
        pair_messages = 1
        if i in edge_keys:
            for j in node_edge_dict[i]:
                pair_messages *= (1 - beta * edge_p[(j,i)])

        tri_messages  = 1
        if i in tri_keys:
            for (k, r) in node_tri_dict[i]:
                k, r = sorted([k, r])
                tri_messages *=  (1-beta * ii_p[(k,r,i)])\
                                *(1-beta * ii_p[(k,r,i)])\
                                *(1 - beta * is_p[(k,r,i)])\
                                *(1 - beta * is_p[(r,k,i)])
        meta_values[i] = pair_messages * tri_messages
    return meta_values

@njit()
def calculateCavityMessagesEdge(place_holder, node_edge_dict, node_tri_dict, edge_p, ii_p, is_p, beta, edge_keys, tri_keys):
    cavity_messages_one = place_holder
    for (j, i), _ in cavity_messages_one.items():
        pair_messages = 1
        if j in edge_keys:
            for k in node_edge_dict[j]:
                if k != i:
                    pair_messages *= (1-beta * edge_p[(k, j)])
        
        tri_messages = 1
        if j in tri_keys:
            for (k, r) in node_tri_dict[j]:
                k, r = sorted([k, r])
                tri_messages *= (1-beta* ii_p[(k,r,j)])\
                                *(1-beta* ii_p[(k,r,j)])\
                                * (1 - beta * is_p[(r,k,j)])\
                                * (1 - beta * is_p[(k,r,j)])
        
        cavity_messages_one[(j,i)] = pair_messages * tri_messages

    return cavity_messages_one

@njit()
def calculateCavityMessagesTri(place_holder, node_edge_dict, node_tri_dict, edge_p, ii_p, is_p, beta, edge_keys, tri_keys):
    cavity_messages_tri = place_holder.copy()
    for (k, r, i) in cavity_messages_tri.keys():
        pair = 1
        if k in edge_keys:
            for n in node_edge_dict[k]:
                pair *= (1 - beta * edge_p[(n, k)])
        tri = 1
        if k in tri_keys:
            for (n, m) in node_tri_dict[k]:
                if n not in [i, r] or m not in [i, r]:
                    n, m = sorted([n, m])
                    tri *= (1-beta* ii_p[(n,m,k)])\
                           *(1-beta* ii_p[(n,m,k)])\
                           * (1 - beta * is_p[(n, m, k)])\
                           * (1 - beta * is_p[(m, n, k)])
        cavity_messages_tri[(k, r, i)] = pair * tri
    
    return cavity_messages_tri
    

@njit()
def SIS_DMP_base(N, gamma, beta, node_p, edge_p, ii_p, is_p, node_edge_dict, node_tri_dict, edge_keys, tri_keys):

    messages = calculateMessages(N, node_edge_dict, node_tri_dict, edge_p, ii_p, is_p, beta, edge_keys, tri_keys)

    place_holder = edge_p.copy()
    cavity_messages_edge = calculateCavityMessagesEdge(place_holder, node_edge_dict, node_tri_dict, edge_p, ii_p, is_p, beta, edge_keys, tri_keys)

    place_holder = is_p.copy()
    cavity_messages_tri = calculateCavityMessagesTri(place_holder, node_edge_dict, node_tri_dict, edge_p, ii_p, is_p, beta, edge_keys, tri_keys)

    new_node_p = node_p.copy()
    for i, v in node_p.items():
        new_node_p[i]  = node_p[i] * (1 - gamma) + (1 - node_p[i]) * (1 - messages[i])
    
    new_edge_p = edge_p.copy()
    for (j, i), v in new_edge_p.items():
        new_edge_p[(j,i)] = edge_p[(j,i)] * (1 - gamma) + (1 - edge_p[(j,i)]) * (1 - cavity_messages_edge[(j, i)])
    
    new_ii_p = ii_p.copy()
    for (k, r, i), v in new_ii_p.items():
        new_ii_p[(k, r, i)] =     ii_p[(k,r,i)] * (1 - gamma)**2 \
                                + is_p[(r,k,i)] * (1-gamma) * (1 - cavity_messages_tri[(k, r, i)]*(1-beta))\
                                + is_p[(k,r,i)] * (1-gamma) * (1 - cavity_messages_tri[(r, k, i)]*(1-beta))\
                                + (1-ii_p[(k,r,i)]-is_p[(r,k,i)]-is_p[(k,r,i)]) * (1-cavity_messages_tri[(k, r, i)]) * (1-cavity_messages_tri[(r, k, i)])
    
    new_is_p = is_p.copy()
    for (k,r,i), v in new_is_p.items():
        k1, r1 = sorted([k, r])
        new_is_p[(k, r, i)] =   is_p[(k, r,i)] * (1-gamma) * cavity_messages_tri[(r, k, i)]*(1-beta) \
                               + ii_p[(k1,r1,i)] * gamma * (1-gamma) \
                               + is_p[(r,k,i)] * gamma * (1-cavity_messages_tri[(k, r, i)]*(1-beta)) \
                               + (1-ii_p[(k1,r1,i)]-is_p[(r,k,i)]-is_p[(k,r,i)]) * (1-cavity_messages_tri[(k, r, i)]) * cavity_messages_tri[(r, k, i)]

    return new_node_p, new_edge_p, new_ii_p, new_is_p
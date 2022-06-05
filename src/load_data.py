# -*- encoding: utf-8 -*-
'''
@File    :   load_data.py
@Time    :   2022/05/30 18:44:37
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import random
import networkx as nx
import numpy as np
from collections import defaultdict

def import_random_er(n:int, k:float):
    p = k/(n-1)
    G = nx.erdos_renyi_graph(n, p)
    return post_process(G)

def import_random_sw(n:int, m:int, p:float=0.3):
    G = nx.watts_strogatz_graph(n, k=m, p=p)
    return post_process(G)

def import_random_sf(n:int, m:int):
    G = nx.barabasi_albert_graph(n, m=m)
    return post_process(G)

def post_process(G:nx.Graph):
    n = G.number_of_nodes()
    for node in G.nodes():
        if G.degree(node) == 0:
            target = np.random.choice(range(n), 1)
            G.add_edges_from([(node, target[0])])

    # rearange nodes' indexes
    N = G.order()
    node_relabel_map = {node:i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, node_relabel_map)

    cliques2 = [x for x in list(nx.enumerate_all_cliques(G)) if len(x) == 2]
    cliques3 = [x for x in list(nx.enumerate_all_cliques(G)) if len(x) == 3] 

    # random select non-intersection triangles
    random.shuffle(cliques3)
    occupied_edges = []
    selected_triangles = []
    for (i,j,k) in cliques3:
        i,j,k = sorted([i,j,k])
        if (i,j) in occupied_edges or (i,k) in occupied_edges or (j,k) in occupied_edges:
            continue
        else:
            selected_triangles.append((i,j,k))
            occupied_edges.extend([(i,j), (i,k), (j,k)])
    
    # the edges not in selected triangles
    selected_edges = []
    for (i,j) in cliques2:
        i,j = sorted([i,j])
        if (i,j) in occupied_edges:
            continue
        else:
            selected_edges.append((i,j))
    
    # build neighboring dict
    node_neig_dict = defaultdict(set)
    node_edge_dict = defaultdict(set)
    node_tri_dict  = defaultdict(set)

    for (u,v) in cliques2:
        node_neig_dict[v].add(u)
        node_neig_dict[u].add(v)

    for (u,v) in selected_edges:
        node_edge_dict[v].add(u)
        node_edge_dict[u].add(v)

    for (u,v,w) in selected_triangles:
        node_tri_dict[u].add((v,w))
        node_tri_dict[v].add((u,w))
        node_tri_dict[w].add((u,v))

    node_neig_dict = {k:np.array(list(v)) for k,v in node_neig_dict.items()}
    node_edge_dict = {k:np.array(list(v)) for k,v in node_edge_dict.items()}
    node_tri_dict  = {k:np.array(list(v)) for k,v in node_tri_dict.items()}

    return node_neig_dict, node_edge_dict, node_tri_dict

    

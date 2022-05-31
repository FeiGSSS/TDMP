# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/05/31 10:34:29
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import argparse
from unittest import result
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool

from src.utils import parser_mc_results, parser_results
from src.load_data import import_random_er, import_random_sw
from methods.MC.core import SIS_MC
from methods.ELE.core import SIS_ELE

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Message passing with triangles")
    parser.add_argument("--net", type=str, default="er", help="the type of structure of networks")
    parser.add_argument("--net_k", type=float, default=6, help="the average degree in er net")
    parser.add_argument("--net_m", type=int, default=6, help="the parameter of node in small world")
    parser.add_argument("--n", type=int, default=500, help="the number of nodes")
    parser.add_argument("--cpu_core", type=int, default=50, help="the number of cpu cores to be used")
    parser.add_argument("--num_mc", type=int, default=100)
    parser.add_argument("--MC", action="store_true")
    parser.add_argument("--ELE", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.1, help="the recover probability of every node")
    parser.add_argument("--p0", type=float, default=0.5, help="the percent of initial infencted nodes")
    parser.add_argument("--tmax", type=int, default=1000, help="the maximum number of steps")
    parser.add_argument("--steady", type=bool, default=True)
    args = parser.parse_args()

    if args.net == "er":
        assert args.net_k is not None
        node_neig_dict, node_edge_dict, node_tri_dict = import_random_er(args.n, k=args.net_k)
    elif args.net == "sw":
        assert args.net_m is not None
        node_neig_dict, node_edge_dict, node_tri_dict = import_random_sw(args.n, m=args.net_m)
    else:
        raise NotImplementedError

    degree = sum([len(v) for v in node_neig_dict.values()])/args.n
    lambdas = np.linspace(0.5, 1.5, 20)
    betas = lambdas * args.gamma / degree
    I0 = np.random.choice(np.arange(args.n), size=int(args.n*args.p0), replace=False)

    if args.MC:
        rhos = {}
        for beta in tqdm(betas):
            rhos[beta] = SIS_MC(args.n, beta, args.gamma, node_neig_dict, args.tmax, I0, args.cpu_core, args.num_mc, True)
        mc_rho = parser_mc_results(rhos, cut=True)
        print(mc_rho)

    if args.ELE:
        pool = Pool(min([args.cpu_core, len(betas)]))
        results = []
        for beta in betas:
            results.append(pool.apply_async(SIS_ELE, (args.n, beta, args.gamma, node_neig_dict, args.tmax, I0)))
        pool.close()
        pool.join()

        res = [r.get() for r in results]
        rho = {}
        for b, r in res:
            rho[b] = r
        rho = parser_results(rho)
        print(rho)
    



    
    
    

    


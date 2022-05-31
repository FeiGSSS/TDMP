# -*- encoding: utf-8 -*-
'''
@File    :   core.py
@Time    :   2022/05/31 11:02:48
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

from methods.MC.base import SIS_MC_batch
from multiprocessing import Pool

def SIS_MC(N, beta, gamma, node_neig_dict, tmax,I0, cpu, mc_num, steady=True):

    batch_size = int(mc_num / cpu)
    
    pool = Pool(cpu)
    results = []
    for p in range(cpu):
        results.append(pool.apply_async(SIS_MC_batch, (batch_size, p, N, beta, gamma, node_neig_dict, tmax, I0, steady)))
    pool.close()
    pool.join()

    rhos = []
    for r in results:
        rhos.extend(r.get())
    
    return rhos
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/05/31 11:28:32
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import numpy as np

def parser_mc_results(rho, cut:bool=True):
    rhos_array = np.vstack([x[1] for x in sorted(zip(rho.keys(), rho.values()), key=lambda x:x[0])]).T

    if cut:
        cut_point = min(np.argwhere(np.count_nonzero(rhos_array, axis=0)>1))[0]
        cut_rhos_array = []

        for rhos in rhos_array:
            clean_rhos = []
            for i, rr in enumerate(rhos):
                if i<cut_point:
                    clean_rhos.append(rr)
                elif rr==0:
                    clean_rhos.append(np.nan)
                else:
                    clean_rhos.append(rr)
            cut_rhos_array.append(clean_rhos) 

        cut_rhos_array = np.array(cut_rhos_array)
        avg_rhos = np.nanmean(cut_rhos_array, axis=0)
    else:
        avg_rhos = np.mean(rhos_array, axis=0)
        
    return avg_rhos

def parser_results(rho):
    return np.array([x[1] for x in sorted(zip(rho.keys(), rho.values()), key=lambda x:x[0])])

import os
def checkFolder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

import pickle as pkl
def save(path, data):
    with open(path, "wb") as f:
        pkl.dump(data, f)

def load(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data
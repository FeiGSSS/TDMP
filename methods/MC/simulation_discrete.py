
import numpy as np
from multiprocessing import Pool

def SIS_discrete(G, tau, gamma, initial_infecteds, tmax):
    status = {node:0 for node in G.nodes()}
    for node in initial_infecteds:
        status[node] = 1
    
    logs = [status.copy()]
    for _ in range(tmax):
        new_status = status.copy()
        for node in G.nodes():
            eventp = np.random.random_sample()
            if status[node] == 1: #infected
                if eventp < gamma:
                    new_status[node] = 0
            elif status[node] == 0: # suspected
                assert not G.is_directed()
                neighbors = G.neighbors(node)
                infected_neighbors = [v for v in neighbors if status[v] == 1]
                if eventp < 1 - (1 - tau) ** len(infected_neighbors):
                    new_status[node] = 1
            else:
                return
        logs.append(new_status.copy())
        status = new_status.copy()
    return logs

def sub_task(batch_size, args):
    G, tau, gamma, initial_infecteds, tmax = args

    res = []
    for _ in range(batch_size):
        logs = SIS_discrete(G, tau, gamma, initial_infecteds, tmax)
        rho = [sum(d.values())/G.order() for d in logs]
        res.append(rho)
    return res

def SIS_discrete_iter(iterations, pool_num, args):
    pool = Pool(pool_num)
    batch_size = int(iterations // pool_num)
    
    rho = []
    for _ in range(pool_num):
        pool.apply_async(sub_task, (batch_size, args), callback=rho.extend)
    pool.close()
    pool.join()
    rho = np.array(rho) # [iteration, T]
    return np.mean(rho, axis=0).squeeze()
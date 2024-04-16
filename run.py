import numpy as np
import math
import os
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import collections
import random
import numpy.linalg as LA
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize, NonlinearConstraint
import math
from utils import *
from ConDuel import *
from R_ConUCB import *
import argparse

parser = argparse.ArgumentParser(description='ConDuel')
parser.add_argument('--dataset', '-d', default='synthetic10', type=str)
parser.add_argument('--pool', '-p', default=50, type=int)
parser.add_argument('--bt', '-b', default='lin_10', type=str)
parser.add_argument('--horizon', default=5000, type=int)
parser.add_argument('--alpha', '-a', default=1.5, type=float)
parser.add_argument('--type', '-t', default='cond', type=str)
args = parser.parse_args()
dataset = args.dataset
horizon = args.horizon
bt = {
    'log'    : lambda t: int(math.log(t + 1)),
    'log_5'  : lambda t: 5 * int(math.log(t + 1)),
    'log_10' : lambda t: 10*int(math.log(t + 1)),
    'log_20' : lambda t: 20*int(math.log(t + 1)),
    'lin'    : lambda t: int(t/50),
    'lin_5'  : lambda t: 5*int(t/50),
    'lin_10' : lambda t: 10* int(t/50),
    'lin_20' : lambda t: 20* int(t/50),
    'lin_50' : lambda t: 50*int(t/50)
}
bt_func = bt[args.bt]
pool_index_list = np.load('index_list_%d_50k.npy'%args.pool)
arms = np.load(os.path.join(dataset, 'arms.npy'))
suparms = np.load(os.path.join(dataset, 'suparms.npy'))
users = np.load(os.path.join(dataset, 'users.npy'))
B = np.load(os.path.join(dataset, 'barycentric.npy'))
dim  = arms.shape[1]
    
    
algo_names = ["ConDuel-Random", "ConDuel-Maxinp", "ConDuel", "MaxInp", "Random_opt", "Rconucb-PosNeg", "Rconucb-Diff"]  
alpha = args.alpha
conf = {"lamb": 5, "sigma": 0.05, "arm_norm_ub": 1, 'eta':0.05, "param_norm_ub": 2, "length": 50,
            "bt": bt_func}

noconf = {"lamb": 5, "sigma": 0.05, "arm_norm_ub": 1, 'eta':0.05, "param_norm_ub": 2, "length": 50}

rconucb_conf = {'lamb': 0.5, 'tilde_lamb': 1, 'alpha': 0.25,'tilde_alpha':0.25}


def main():
    # pth = os.path.join('results', str(args.pool))
    # if not os.path.exists(pth):
    #     os.mkdir(pth)
    pth = os.path.join('rebuttal/duel-5k', dataset)
    if not os.path.exists(pth):
        os.mkdir(pth)
    pth = os.path.join(pth, args.bt)
    if not os.path.exists(pth):
        os.mkdir(pth)
    start = time.time()
    users_pool = [(i,pth,users) for i in range(20)]
    with Pool(processes = 20) as pool:
        results = pool.starmap(sum_single_user, (users_pool))
        pool.close()
        pool.join()
    end = time.time()

    print("total time:", end - start)

def run_user(user, arms, suparms, B, algo_names, algo_noconf,  rconucb_conf, buget_func,
             pool_index_list, horizon):

    dim = len(user)
    algos_nocon = {
        "MaxInp": Con_Duel(dim, algo_noconf,alpha, "maxinp","random", user),
        "Random_opt": Con_Duel(dim, algo_noconf,alpha, "random_optimal", "random", user)
    }
    algos_con = {
        "ConDuel-Random": Con_Duel(dim, conf, alpha,"con_duel", "random",  user),
        "ConDuel-Maxinp": Con_Duel(dim, conf, alpha, "con_duel", "maxinp", user),
        "ConDuel": Con_Duel(dim, conf, alpha, "con_duel", "barycentric spanner", user)
    }
    algos_rconucb = {
        "Rconucb-PosNeg": R_ConUCB(dim, rconucb_conf, "pos&neg", user),
        "Rconucb-Diff": R_ConUCB(dim, rconucb_conf, "diff", user)
    }
    # run our conversational dueling bandit algorithms

    algos_regret = {}
    algos_time = {}
    algos_armtime = {}
    algos_suparmtime = {}
    
    for algoname in algo_names:
        # initialize
        algos_regret[algoname] = []
        algos_time[algoname] = []
        algos_armtime[algoname] = []
        algos_suparmtime[algoname] = []
        algos_weakregret[algoname] = []
        algos_strongregret[algoname] = []

    for algoname, algo in algos_nocon.items():
        print("noncon")
        if algoname in algo_names:
            algo.init(arms)
            # print('inited')
            start = time.time()
            for i in range(horizon):
                # print('itr')
                arm_pool = arms[pool_index_list[i]]
                a1, a2, arm_diff = algo.choose_arm_pair(arm_pool)
                algo.update_parameters(np.array([a1, a2]))
                algo.update_reward(a1, a2, arm_pool)
            end = time.time()
            regret_tmp = algo.regrets
            armtime_tmp = algo.t_arm
            algos_suparmtime[algoname] = 0
            algos_armtime[algoname] = armtime_tmp
            algos_regret[algoname] = [sum(regret_tmp[0:i]) for i in range(len(regret_tmp))]
            algos_weakregret[algoname] = [sum(regret_weak[0:i]) for i in range(len(regret_tmp))]
            algos_strongregret[algoname] = [sum(regret_strong[0:i]) for i in range(len(regret_tmp))]
            algos_time[algoname] = end - start

    # run our conversational dueling bandit algorithms
    for algoname, algo in algos_con.items():
        print("con")
        if algoname in algo_names:
            start = time.time()
            algo.init(arms)
            for i in range(horizon):
                if buget_func(i + 1) - buget_func(i) > 0:
                    conv_times = int(buget_func(i + 1) - buget_func(i))
                    for j in range(conv_times):
                        s1, s2, suparm_diff = algo.choose_suparm_pair(suparms, B)
                        algo.update_parameters(np.array([s1, s2]))
                arm_pool = arms[pool_index_list[i]]
                a1, a2, arm_diff = algo.choose_arm_pair(arm_pool)
                algo.update_parameters(np.array([a1,a2]))
                algo.update_reward(a1, a2, arm_pool)
            end = time.time()
            regret_tmp = algo.regrets
            armtime_tmp = algo.t_arm
            suparmtime_tmp = algo.t_suparm
            algos_armtime[algoname] = armtime_tmp
            algos_suparmtime[algoname] = suparmtime_tmp
            algos_regret[algoname] = [sum(regret_tmp[0:i]) for i in range(len(regret_tmp))]
            algos_time[algoname] = end - start

    for algoname, algo in algos_rconucb.items():
        print("rcon")
        if algoname in algo_names:
            start = time.time()
            for i in (range(horizon)):
                arm_pool = arms[pool_index_list[i]]
                if buget_func(i + 1) - buget_func(i) > 0:
                    conv_times = int(buget_func(i + 1) - buget_func(i))
                    for j in range(conv_times):
                        picked_x, duel_x = algo.choose_suparm_pair(suparms, arm_pool)
                        if np.dot(picked_x - duel_x, user) > 0:
                            algo.update_suparm_pair(picked_x, duel_x)
                        else:
                            algo.update_suparm_pair(duel_x, picked_x)
                a = algo.choose_arm(arm_pool)
                algo.update_parameters(a)
            end = time.time()
            regret_tmp = algo.regrets
            armtime_tmp = algo.t_arm
            suparmtime_tmp = algo.t_suparm
            algos_armtime[algoname] = armtime_tmp
            algos_suparmtime[algoname] = suparmtime_tmp
            algos_regret[algoname] = [sum(regret_tmp[0:i]) for i in range(len(regret_tmp))]
            algos_time[algoname] = end - start
            
    sum_user = {"algos_regret": algos_regret, "algos_time": algos_time,
                "algos_armtime": algos_armtime, "algos_suparmtime": algos_suparmtime, 
                }
    return sum_user


def sum_single_user(i,pth,users):
    #print(i)
    user_pth = os.path.join(pth, 'single_user_'+ str(i) + '.npy')
    if os.path.isfile(user_pth):
        return 0
    print(i)
    user = users[i]
    
    results = run_user(user, arms, suparms, B, algo_names, 
                       noconf, rconucb_conf, conf['bt'], pool_index_list, horizon)
    np.save(user_pth, results)
    return results


if __name__ == '__main__':
    main()

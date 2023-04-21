import numpy as np
import math
from basic_func import *
import matplotlib.pyplot as plt
import time, os
from multiprocessing import Pool

class R_ConUCB:
    def __init__(self, dim, conf, suparm_strategy, theta_star):

        self.dim = dim
        self.theta_star = theta_star
        self.lamb = conf["lamb"]
        self.tilde_lamb = conf["tilde_lamb"]
        self.alpha = conf["alpha"]
        self.tilde_alpha = conf["tilde_alpha"]
        # self.sigma = conf["sigma"]

        self.suparm_strategy = suparm_strategy
        self.theta = np.zeros((self.dim,1))
        self.tilde_theta = np.zeros((self.dim,1))
        self.theta_norm = np.linalg.norm(theta_star)


        self.M = (1-self.lamb) * np.identity(n=self.dim)
        self.Minv = 1 / (1-self.lamb) * np.identity(n=self.dim)
        self.Y = np.zeros((self.dim, 1))
        self.tilde_M = self.tilde_lamb * np.identity(n=self.dim)
        self.tilde_Minv = 1/self.tilde_lamb * np.identity(n=self.dim)
        self.tilde_Y = np.zeros((self.dim, 1))

        self.rewards = []
        self.regrets = []
        self.ctr = 0

    def update_Minv(self, Minv, x):

        tmp_a = np.dot(np.outer(np.dot(Minv, x), x), Minv)
        tmp_b = 1 + np.dot(np.dot(x.T, Minv), x)
        new_Minv = Minv - tmp_a / tmp_b
        return new_Minv

    """key-term module"""

    def get_credit(self, tilde_x, arm_pool):

        a = np.dot(np.dot(np.dot(arm_pool, self.Minv), self.tilde_Minv), tilde_x)
        b = 1 + weighted_norm(tilde_x, self.tilde_Minv)**2
        norm_M = np.linalg.norm(a)
        return norm_M * norm_M/b

    def choose_suparm(self, suparms, arm_pool):

        values = [self.get_credit(x, arm_pool) for x in suparms]
        suparm = suparms[np.argmax(values)]
        return suparm

    def choose_suparm_pair(self, suparms, arm_pool):

        picked_x = self.choose_suparm(suparms, arm_pool)
        duel_x = suparms[np.argmax([self.get_credit(x-picked_x, arm_pool) for x in suparms])]
        return picked_x, duel_x

    def update_sup_parameters(self, tilde_x, tilde_y):

        self.tilde_Minv = self.update_Minv(self.tilde_Minv, tilde_x)
        self.tilde_M += np.outer(tilde_x, tilde_x)
        self.tilde_Y += (tilde_x * tilde_y).reshape(self.dim,1)
        self.tilde_theta = np.dot(self.tilde_Minv, self.tilde_Y)
        self.theta = np.dot(self.Minv, self.Y + (1 - self.lamb) * self.tilde_theta)

    def update_suparm_pair(self, picked_x, duel_x):

        if self.suparm_strategy == "pos&neg":
            self.update_sup_parameters(picked_x,1)
            self.update_sup_parameters(duel_x, 0)

        if self.suparm_strategy == "diff":
            diff = picked_x - duel_x
            self.update_sup_parameters(diff, 1)


    """item module"""

    def get_prob(self, x):

        mean = np.dot(x, self.theta)
        var1 = weighted_norm(x,self.Minv)
        tmp = np.dot(x,np.dot(np.dot(self.Minv,self.tilde_Minv),self.Minv))
        var2 = np.sqrt(np.dot(tmp,x))
        uncertainty = self.lamb*self.alpha * var1 + (1-self.lamb)*self.tilde_alpha * var2
        #print(uncertainty)
        ucb = mean + uncertainty
        return ucb

    def choose_arm(self, arm_pool):

        ucb_arms = [self.get_prob(x) for x in arm_pool]
        arm = arm_pool[np.argmax(ucb_arms)]
        reward = sigmoid(np.dot(arm, self.theta_star))
        #reward = np.random.binomial(1,np.dot(arm, self.theta_star))
        self.rewards.append(reward)
        self.regrets.append(sigmoid(np.max(arm_pool @ self.theta_star)) - reward)
        return arm

    def update_parameters(self, x):

        prob = np.dot(x, self.theta_star)
        #prob = max(0.0, prob)
        #prob = min(1.1, prob)
        #y = np.random.binomial(1, prob)
        y = np.random.binomial(1, sigmoid(prob))
        self.Minv = self.update_Minv(self.Minv, np.sqrt(self.lamb) * x)
        self.M += self.lamb * np.outer(x, x)
        self.Y += self.lamb * (x * y).reshape(self.dim,1)
        self.theta = np.dot(self.Minv, self.Y + (1 - self.lamb) * self.tilde_theta)


def run_user_rconucb(user, arms, suparms,conf,pool_index_list, horizon):

    algo_names = ["Rconucb-PosNeg","Rconucb-Diff"]
    dim = len(user)
    algos_regret = {}
    for algoname in algo_names:
        algos_regret[algoname] = []
    algos = {
        "Rconucb-PosNeg": R_ConUCB(dim, conf, "pos&neg", user),
        "Rconucb-Diff": R_ConUCB(dim, conf, "diff", user)
    }
    # run our conversational dueling bandit algorithms
    buget_func = conf["bt"]
    for algoname, algo in algos.items():
        for i in range(horizon):
            arm_pool = arms[pool_index_list[i]]
            if buget_func(i + 1) - buget_func(i) > 0:
                conv_times = int(buget_func(i + 1) - buget_func(i))
                for j in range(conv_times):
                    picked_x, duel_x = algo.choose_suparm_pair(suparms, arm_pool)
                    if np.dot(picked_x - duel_x,user) >0:
                        algo.update_sup_parameters(picked_x, duel_x)
                    else:
                        algo.update_sup_parameters(duel_x, picked_x)
            a = algo.choose_arm(arm_pool)
            algo.update_parameters(a)
        regret_tmp = algo.regrets
        cum_regret = [sum(regret_tmp[0:i]) for i in range(len(regret_tmp))]
        algos_regret[algoname] = cum_regret
    sum_user = {"algos_regret": algos_regret}
    return sum_user


if __name__ == "__main__":
    pth = "/data/ysh/project_uai/data/movielens"
    arms = np.load(os.path.join(pth, "arms.npy"))
    suparms = np.load(os.path.join(pth, "suparms.npy"))
    B = np.load(os.path.join(pth, "barycentric.npy"))
    dim = arms.shape[1]
    pool_index_list = np.load('/data/ysh/project_k/syn_poolsize_50.npy')
    horizon = 5000

    conf = {"lamb": 0.5, "tilde_lamb": 1, "sigma": 0.05, "alpha": 0.25,
            "bt": lambda t: int(t / 50), "tilde_alpha": 0.25}
    users = np.load(os.path.join(pth, "users.npy"))
    user = users[0]

    sum_user = run_user_rconucb(user, arms, suparms,conf,pool_index_list, horizon)
    np.save("/data/ysh/test_rconucb_movie", sum_user)
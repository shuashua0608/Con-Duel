import numpy as np
import math
from basic_func import *
import time, os
from multiprocessing import Pool
from RConUCB import R_ConUCB
class Con_Duel: #param_strategy
    def __init__(self, dim, conf, arm_strategy, suparm_strategy, param_strategy,theta_star):

        self.dim = dim
        self.length = conf["length"]
        self.lamb = conf["lamb"]
        self.eta = conf["eta"]
        self.alpha = conf["alpha"]
        self.arm_norm_ub = conf["arm_norm_ub"]
        self.param_norm_ub = conf["param_norm_ub"]
        self.arm_strategy = arm_strategy
        self.suparm_strategy = suparm_strategy
        self.param_strategy = param_strategy
        self.theta_star = theta_star

        self.kappa = get_kappa(self.arm_norm_ub, self.param_norm_ub)
        self.M = self.lamb * self.kappa * np.identity(n=self.dim)
        self.Minv = 1 / (self.lamb * self.kappa) * np.identity(n=self.dim)
        self.theta = np.random.normal(0, 1, (self.dim,))
        self.X = []
        self.y = []
        self.rewards = []
        self.regrets = []
        self.ctr = 0
        self.t_arm = 0
        self.t_suparm = 0

    def init(self, arms):
        for i in range(self.length):
            row = np.arange(arms.shape[0])
            np.random.shuffle(row)
            a1 = arms[row[0]]
            a2 = arms[row[1]]
            d_i = a1 - a2
            r_i = sigmoid(np.dot(d_i, self.theta_star))
            self.X.append(d_i.reshape(self.dim))
            self.y.append(np.random.binomial(1, r_i))
            self.M = self.M + np.outer(d_i, d_i)
            self.Minv = self.update_Minv(self.Minv, d_i)

    def update_Minv(self, Minv, diff):

        tmp_a = np.dot(np.outer(np.dot(Minv, diff), diff), Minv)
        tmp_b = 1 + np.dot(np.dot(diff.T, Minv), diff)
        new_Minv = Minv - tmp_a / tmp_b
        return new_Minv

    def update_parameters(self, diff):

        prob = sigmoid(np.dot(diff, self.theta_star))
        yt = np.random.binomial(1, prob)
        self.y.append(yt)
        self.X.append(diff.reshape(self.dim))
        self.Minv = self.update_Minv(self.Minv, diff)

        if self.param_strategy == "quick":
           self.theta += self.eta/np.sqrt(self.ctr+1)*(yt - prob) * diff

        if self.param_strategy == "normal":
           if self.ctr % 5 == 0 or len(self.regrets) <200:
              self.theta = solve_MLE(self.X, self.y, 1/self.lamb)
        """
        #self.theta += self.eta / np.sqrt(self.ctr+1) * (yt - prob) * diff
        if self.ctr % 10 == 0 or len(self.regrets) < 200:
           self.theta = solve_MLE(self.X, self.y, 1 / self.lamb)
        """
        self.ctr += 1

    def choose_arm_pair(self, arm_pool):

        start = time.time()
        C_t = []
        arm_1 = arm_2 = arm_diff = None

        for arm in arm_pool:
            arm_ucb = [np.dot(arm - arm_other, self.theta)+self.alpha * weighted_norm(arm - arm_other, self.Minv)
                       for arm_other in arm_pool]
            if all(i >= 0 for i in arm_ucb):
                C_t.append(arm)

        if self.arm_strategy == "random_optimal":
            arm_1 = C_t[random.randint(0, len(C_t) - 1)]
            arm_2 = C_t[random.randint(0, len(C_t) - 1)]
            arm_diff = arm_1 - arm_2

        if self.arm_strategy == "maxinp":
            ucb_record = []
            arm_record = []
            for arm in C_t:
                C_temp = [weighted_norm(arm - arm_other, self.Minv) for arm_other in C_t]
                arm_record.append(C_t[np.argmax(C_temp)])
                ucb_record.append(max(C_temp))
            arm_1 = C_t[np.argmax(ucb_record)]
            arm_2 = arm_record[np.argmax(ucb_record)]
            arm_diff = arm_1 - arm_2

        if self.arm_strategy == "con_duel":
            arm_1 = C_t[random.randint(0, len(C_t) - 1)]
            uncertainty = [weighted_norm(arm - arm_1, self.Minv) for arm in C_t]
            arm_2 = C_t[np.argmax(uncertainty)]
            arm_diff = arm_1 - arm_2

        end = time.time()
        self.t_arm += end - start
        return arm_1, arm_2, arm_diff

    def update_reward(self, arm_1, arm_2, arm_pool):

        reward1 = sigmoid(np.dot(arm_1, self.theta_star))
        reward2 = sigmoid(np.dot(arm_2, self.theta_star))
        #reward = 0.5 * np.dot(arm_1 + arm_2, self.theta_star)
        reward = 0.5*(reward1 + reward2)
        self.rewards.append(reward)
        self.regrets.append(sigmoid(np.max(arm_pool @ self.theta_star))- reward)

    def choose_suparm_pair(self, suparms, B):

        start = time.time()
        suparm_1 = suparm_2 = suparm_diff = None

        if self.suparm_strategy == "random":
            row = np.arange(suparms.shape[0])
            np.random.shuffle(row)
            ind = row[0:2]
            suparm_1 = suparms[ind[0]]
            suparm_2 = suparms[ind[1]]
            suparm_diff = suparm_1 - suparm_2

        if self.suparm_strategy == "barycentric spanner":
            row = np.arange(B.shape[0])
            np.random.shuffle(row)
            ind = row[0:2]
            suparm_1 = B[ind[0]]
            suparm_2 = B[ind[1]]
            suparm_diff = suparm_1 - suparm_2

        if self.suparm_strategy == "maxinp":

            row = np.arange(suparms.shape[0])
            np.random.shuffle(row)
            suparm_1 = suparms[row[0]]
            diff_norm = [weighted_norm(x- suparm_1, self.Minv) for x in suparms]
            suparm_2 = suparms[np.argmax(diff_norm)]
            suparm_diff = suparm_1 - suparm_2

        end = time.time()
        self.t_suparm += end - start
        return suparm_diff


def run_user(user, arms, suparms, B, algo_names, algo_conf, algo_noconf,  rconucb_conf,
             param_strategy, pool_index_list, horizon, p):

    dim = len(user)
    algos_nocon = {
        "MaxInp": Con_Duel(dim, algo_noconf,"maxinp","random",param_strategy,user),
        "Random_opt": Con_Duel(dim, algo_noconf,"random_optimal", "random", param_strategy, user)
    }

    algos_con = {
        "ConDuel-random": Con_Duel(dim, algo_conf, "con_duel", "random", param_strategy,user),
        "ConDuel-Maxinp": Con_Duel(dim, algo_conf, "con_duel", "maxinp",param_strategy,user),
        "ConDuel": Con_Duel(dim, algo_conf,"con_duel", "barycentric spanner",param_strategy, user)
    }

    algos_rconucb = {
        "Rconucb-PosNeg": R_ConUCB(dim, rconucb_conf, "pos&neg", user),
        "Rconucb-Diff": R_ConUCB(dim, rconucb_conf, "diff", user)
    }
    # run our conversational dueling bandit algorithms

    algos_regret = {}
    algos_time = {}
    for algoname in algo_names:
        # initialize
        algos_regret[algoname] = []
        algos_time[algoname] = []

    for algoname, algo in algos_nocon.items():
        if algoname in algo_names:
            algo.init(arms)
            start = time.time()
            for i in range(horizon):
                arm_pool = arms[pool_index_list[i]]
                a1, a2, arm_diff = algo.choose_arm_pair(arm_pool)
                algo.update_parameters(arm_diff)
                algo.update_reward(a1, a2, arm_pool)
            end = time.time()
            regret_tmp = algo.regrets
            cum_regret = [sum(regret_tmp[0:i]) for i in range(len(regret_tmp))]
            algos_regret[algoname] = cum_regret
            algos_time[algoname] = end - start

    #buget_func = algo_conf["bt"]
    # run our conversational dueling bandit algorithms
    for algoname, algo in algos_con.items():
        if algoname in algo_names:
            start = time.time()
            algo.init(arms)
            for i in range(horizon):
                if buget_func(i + 1) - buget_func(i) > 0:
                    conv_times = int(buget_func(i + 1) - buget_func(i))
                    for j in range(conv_times):
                        suparm_diff = algo.choose_suparm_pair(suparms, B)
                        algo.update_parameters(suparm_diff)
            
                arm_pool = arms[pool_index_list[i]]
                a1, a2, arm_diff = algo.choose_arm_pair(arm_pool)
                algo.update_parameters(arm_diff)
                algo.update_reward(a1, a2, arm_pool)
            end = time.time()
            regret_tmp = algo.regrets
            cum_regret = [sum(regret_tmp[0:i]) for i in range(len(regret_tmp))]
            algos_regret[algoname] = cum_regret
            algos_time[algoname] = end - start

    for algoname, algo in algos_rconucb.items():
        if algoname in algo_names:
            start = time.time()
            for i in range(horizon):
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
            cum_regret = [sum(regret_tmp[0:i]) for i in range(len(regret_tmp))]
            algos_regret[algoname] = cum_regret
            algos_time[algoname] = end - start
    sum_user = {"algos_regret": algos_regret, "algos_time": algos_time}

    return sum_user


import numpy as np
import math
from utils import *
import time, os
from multiprocessing import Pool
class Con_Duel: #param_strategy
    def __init__(self, dim, conf, alpha, arm_strategy, suparm_strategy,theta_star):

        self.dim = dim
        self.length = conf["length"]
        self.lamb = conf["lamb"]
        self.eta = conf['eta']
        self.arm_norm_ub = conf["arm_norm_ub"]
        self.param_norm_ub = conf["param_norm_ub"]
        self.alpha = alpha
        #self.param_strategy = conf["param_strategy"]
        self.arm_strategy = arm_strategy
        self.suparm_strategy = suparm_strategy
        self.theta_star = theta_star

        self.kappa = get_kappa(self.arm_norm_ub, self.param_norm_ub)
        self.M = self.lamb * self.kappa * np.identity(n=self.dim)
        self.Minv = 1 / (self.lamb * self.kappa) * np.identity(n=self.dim)
        self.theta = np.random.normal(0, 1, (self.dim,))
        self.X = []
        self.y = []
        self.rewards = []
        self.regrets = []
        self.regrets_s = []
        self.regrets_w = []
        self.ctr = 0
        self.t_arm = 0
        self.t_suparm = 0

    def init(self, arms):
        for i in range(self.length):
            row = np.arange(arms.shape[0])
            np.random.shuffle(row)
            arm_set = arms[row[0:2]]
            a1 = arms[row[0]]
            a2 = arms[row[1]]
            d_i = a1 - a2
            p = choice_output(arm_set, self.theta_star)
            self.y.append(np.array(p,dtype=np.float32))
            self.X.append(np.array(arm_set.reshape(((-1,)))))
            # r_i = sigmoid(np.dot(d_i, self.theta_star))
            # self.X.append(d_i.reshape(self.dim))
            # self.y.append(np.random.binomial(1, r_i))
            
            self.M = self.M + np.outer(d_i, d_i)
            self.Minv = self.update_Minv(self.Minv, d_i)

    def update_Minv(self, Minv, diff):

        tmp_a = np.dot(np.outer(np.dot(Minv, diff), diff), Minv)
        tmp_b = 1 + np.dot(np.dot(diff.T, Minv), diff)
        new_Minv = Minv - tmp_a / tmp_b
        return new_Minv

    def update_parameters(self, arm_set):
        
        a1 = arm_set[0]
        a2 = arm_set[1]
        pt = choice_output(arm_set, self.theta_star)
        self.y.append(np.array(pt))
        self.X.append(np.array(arm_set.reshape((-1,))))
        # prob = sigmoid(np.dot(diff, self.theta_star))
        # yt = np.random.binomial(1, prob)
        # self.y.append(yt)
        # self.X.append(diff.reshape(self.dim))
        self.Minv = self.update_Minv(self.Minv, a1 - a2)

        # if self.param_strategy == "quick":
        #     self.theta += self.eta/np.sqrt(self.ctr+1)*(yt - prob) * diff

#         if self.param_strategy == "normal":
        # if self.ctr % 100 == 0 or len(self.regrets) <100:
        #     self.theta = solve_MLE(self.X, self.y, 1/self.lamb)
        if self.ctr % 100 == 0 or len(self.regrets) <100:
            self.theta = calculate_theta_duel(self.X, self.y, TRAIN_CONFIG = TRAIN_CONFIG2)
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
        reward1 = (np.dot(arm_1, self.theta_star))
        reward2 = (np.dot(arm_2, self.theta_star))
        reward = 0.5*(reward1 + reward2)
        max_reward = max(reward1, reward2)
        min_reward = min(reward1, reward2)
        
        self.rewards.append(reward)
        self.regrets.append((np.max(arm_pool @ self.theta_star))- reward)
        self.regrets_s.append((np.max(arm_pool @ self.theta_star))- min_reward)
        self.regrets_w.append((np.max(arm_pool @ self.theta_star))- max_reward)    
        # reward1 = sigmoid(np.dot(arm_1, self.theta_star))
        # reward2 = sigmoid(np.dot(arm_2, self.theta_star))
        # reward = 0.5*(reward1 + reward2)
        # self.rewards.append(reward)
        # self.regrets.append(sigmoid(np.max(arm_pool @ self.theta_star))- reward)

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
        return suparm_1, suparm_2, suparm_diff

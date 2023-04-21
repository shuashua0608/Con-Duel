from basic_func import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as LA
import argparse
import collections
import os
import random

from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.misc import derivative
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from multiprocessing import Pool
import time
from functools import partial
from matplotlib.pyplot import MultipleLocator
import matplotlib.colors as mcolors

####
algo_names = ["Rconucb-PosNeg","Rconucb-Diff","Random_opt","MaxInp", "ConDuel-random", "ConDuel-maxinp", "ConDuel-bs"]
syn_sum = np.load("/data/ysh/project_uai/data/synthetic/rebuttal/syn_linsum1.npy", allow_pickle=True)
syn = np.load("/data/ysh/project_uai/data/synthetic/rebuttal/syn_lin1.npy", allow_pickle=True)
_,_,regret_std = sum_users(syn, algo_names)
###

syn1_sum = np.load("/data/ysh/project_uai/data/synthetic/rebuttal_new/linsum1.npy", allow_pickle=True)
syn1 = np.load("/data/ysh/project_uai/data/synthetic/rebuttal_new/lin1.npy", allow_pickle=True)

#


MEDIUM_SIZE = 10
BIGGER_SIZE = 14
# "ConDuel-random", "ConDuel-maxinp",
#algo_names = ["ConDuel-random", "ConDuel-maxinp", "ConDuel-bs", "MaxInp", "Random_opt","Rconucb-PosNeg","Rconucb-Diff"]

#algo_names_select = ["ConDuel-bs", "Rconucb-PosNeg", "Rconucb-Diff", "MaxInp"]
# reg_new
#algo_names_select = ["ConDuel-bs", "Rconucb-PosNeg", "Rconucb-Diff", "MaxInp"]
# reg_new
#algo_names_select =  ["ConDuel-random", "ConDuel-maxinp", "ConDuel-bs"]
algo_names_select =  algo_names
reg_syn1 = syn1_sum[0]
r0 = reg_syn1["ConDuel-random"]
r1 = reg_syn1["ConDuel-bs"]
#reg_syn1["ConDuel-bs"] = r0
#reg_syn1["ConDuel-random"] = r1

reg_syn1["MaxInp"] = syn_sum[0]["MaxInp"]
reg_syn1["Random_opt"] = syn_sum[0]["Random_opt"]
print(reg_syn1)

plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)

x_major_locator = MultipleLocator(500)
y_major_locator = MultipleLocator(100)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim(-50, 2050)
plt.ylim((-20, 300))

labels = algo_names_select
colors = list(mcolors.TABLEAU_COLORS.keys())

for i, algoname in enumerate(algo_names_select):
    plt.plot(reg_syn1[algoname], label=algo_names_select[i], color=colors[i])
    #plt.fill_between(range(len(reg_syn1[algoname])),(reg_syn1[algoname] - regret_std[algoname]).clip(0, None),
                  # reg_syn1[algoname] + regret_std[algoname], color=colors[i], alpha=0.2)

plt.xlabel('Iterations')
plt.ylabel("Regret")
plt.legend(loc="upper left")
plt.title("b(t) = [t/50]")
plt.grid()

t = ax.yaxis.get_offset_text()
t.set_x(-0.05)

plt.tight_layout()
fig = plt.gcf()

#plt.savefig("/data/ysh/project_uai/data/synthetic/rebuttal/syn_std_all.png", dpi = 500)
plt.show()
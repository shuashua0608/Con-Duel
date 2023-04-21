import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize, NonlinearConstraint
import random
import numpy.linalg as LA

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def weighted_norm(x, A):  ## ||x||_A
    return np.sqrt(np.dot(x, np.dot(A, x)))

def get_kappa(arm_norm_ub, param_norm_ub):
    tmp = 1 / (1 + np.exp(-arm_norm_ub * param_norm_ub))
    tmp = tmp * (1 - tmp)
    kappa = 1 / tmp
    return kappa

def solve_MLE(X, y, lam_inv = 1):

    model = LogisticRegression(penalty='l2', C = lam_inv, fit_intercept=False, solver='saga')
    model.fit(X, y)
    return model.coef_.reshape(-1) # theta_t:  MLE

# sovle MLE variant for arm #
def compute_cost(theta, X, y, lamb, theta_tilde):

    m = len(y)
    h = sigmoid(X @ theta)
    X = np.array(X)
    y = np.array(y)
    J = lamb * (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    J += (1 - lamb) * np.sum((theta - theta_tilde) ** 2) / (2 * m)
    grad = lamb * (X.T @ (h - y)) / m
    grad += (1 - lamb) * (theta - theta_tilde) / m
    return J, grad

def gradient_descent(theta, theta_tilde, X, y, lamb, eta, num_iters):

    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        J_history[i], grad = compute_cost(theta, X, y, lamb, theta_tilde)
        theta -= eta * grad
        # print("theta_tilde: ", LA.norm(theta-theta_tilde))
        # print("theta_star: ", LA.norm(theta-theta_star))
    return theta, J_history

def calculate_theta(theta_tilde, lamb, X, y, eta, num_iters):

    dim = len(theta_tilde)
    theta = np.zeros((dim,))
    theta, J_history = gradient_descent(theta, theta_tilde, X, y, lamb, eta, num_iters)
    return theta

# burning period #
def init(theta_star, arms, lamb, length, kappa):

    dim = len(theta_star)
    y = []
    X = []
    M = np.eye(dim) * lamb * kappa
    for i in range(length):
        row = np.arange(arms.shape[0])
        np.random.shuffle(row)
        a1 = arms[row[0]]
        a2 = arms[row[1]]
        d_i = a1 - a2
        r_i = sigmoid(np.dot(d_i, theta_star))
        X.append(d_i.reshape(dim))
        y.append(np.random.binomial(1, r_i))
        M = M + np.outer(d_i, d_i)

    return X, y, M


def compute_barycentric(suparms):
    n, d = suparms.shape
    M = np.eye(d)
    basis_k = []

    # construct a basis for suparms #
    for i in range(d):

        det_i = []
        M_tmp = M.copy()
        for k in suparms:
            M_tmp[i, :] = k
            det_tmp = np.linalg.slogdet(M_tmp)[1]
            det_i.append(np.exp(det_tmp))
        ki = suparms[np.argmax(det_i)]
        M[i, :] = ki
        basis_k.append(ki)
    basis_array = np.array(basis_k)
    # construct barycentric spanner for suparms #
    for i in range(d):
        xi = basis_array[i]
        basis_i = basis_array.copy()
        for k in suparms:
            basis_i[i, :] = k
            det_k = np.exp(np.linalg.slogdet(basis_i)[1])
            det_i = np.exp(np.linalg.slogdet(basis_array)[1])
            if det_k - det_i > 1e-6:
                xi = k
            else:
                xi = xi
        basis_array[i] = xi

    return basis_array


def sum_users(result, algo_names):
    regret_all = {}
    regret_max = {}
    regret_min = {}
    regret_std = {}

    for algoname in algo_names:
        regret_all[algoname] = []
        regret_max[algoname] = []
        regret_min[algoname] = []
        regret_std[algoname] = []

    for r in result:
        regret = r["algos_regret"]

        for algoname, algo_regret in regret.items():
            regret_all[algoname].append(regret[algoname])

    for algoname, algo in regret_all.items():
        regret_max[algoname] = np.max(algo)
        regret_min[algoname] = np.min(algo)
        regret_std[algoname] = np.std(algo, axis=0)

    return regret_max, regret_min, regret_std
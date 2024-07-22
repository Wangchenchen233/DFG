# -*- coding: utf-8 -*-
# @Time    : 3/27/2024 3:55 PM
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : DFG_SFS.py

import numpy as np
from sklearn.metrics import pairwise_distances

from utility.init_func import orthogonal_
from utility.subfunc import eig_lastk, local_structure_learning


def rank_consistency_feature(X, S, alpha, eta, k, r, random_state):
    '''
    min_{S,M,A,B,F} \sum_{ij} S_{ij}\|X^iAB-X^jAB\|_2^2 + \eta \|S\|_F^2
                    +\alpha \sum_{ij}M_{ij}\|XA_i-XA_j\|_2^2
    s.t. rank(S)=r, 0\leq S\leq 1,
    S=FMF^T,
    F^TF=I_k
    A^TA=I_k,A\geq 0,
    B^TB=I_r
    0\leq M\leq 1
    :param X:
    :param S:
    :param alpha: \sum_{ij}M_{ij}\|XA_i-XA_j\|_2^2
    :param eta: \|S-FMF^T\|_F^2
    :param lambda_s: \|S\|_F^2
    :param k: # feature
    :param r: # class
    :param kn: # knn
    :return:
    '''
    np.random.seed(random_state)
    lambda_a = 1e10  # A^TA=I的约束
    n, d = X.shape
    kfs = int(k / r)

    XX = X.T @ X
    Ls = np.diag(S.sum(0)) - S

    A = np.random.rand(d, k)
    A = orthogonal_(A, random_state)

    np.abs(A, out=A)
    A[A < 1e-6] = 1e-6

    Y, _ = eig_lastk(Ls, r)

    M = np.eye(k)  # init M
    # Lm=np.eye(k)
    F = np.zeros((n, k))  # init F

    max_iter = 30
    for iter_ in range(max_iter):

        # update M
        dist_XA = alpha * pairwise_distances((X @ A).T) ** 2
        dist_fsf = - 2 * eta * F.T @ S @ F
        M = local_structure_learning(kfs, eta, dist_XA, dist_fsf, 1)
        Lm = np.diag(M.sum(0)) - M

        # update F
        if k <= n:
            Uk, _ = eig_lastk(S, k)
            V, _ = eig_lastk(M, k)
            F = Uk @ V.T
        else:
            V, _ = eig_lastk(S, n)
            Un, _ = eig_lastk(M, n)
            F = V @ Un.T

        # update A
        A_down = alpha * XX @ A @ Lm + lambda_a * A @ A.T @ A
        temp = np.divide(lambda_a * A, A_down)
        A = A * np.array(temp)

        temp = np.diag(np.sqrt(np.diag(1 / (np.dot(A.transpose(), A) + 1e-16))))
        A = np.dot(A, temp)

        # update Y
        Y, e_val = eig_lastk(Lm, r)

        fn1 = np.sum(e_val[:r])
        fn2 = np.sum(e_val[:r + 1])
        # print(fn1,fn2)
        if fn1 < 10e-10 and fn2 > 10e-10:
            break
    return A, M

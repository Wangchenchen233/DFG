# -*- coding: utf-8 -*-
# @Time    : 3/27/2024 3:55 PM
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : DFG_UFS.py

import numpy as np
from sklearn.metrics import pairwise_distances

from utility.init_func import orthogonal_
from utility.subfunc import eig_lastk, local_structure_learning


def DFG_UFS(X, S, alpha, eta, lambda_s, k, r, kn, random_state):
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
    lambda_f = lambda_s
    n, d = X.shape
    kfs = int(k / r)

    XX = X.T @ X
    Ls = np.diag(S.sum(0)) - S

    A = np.random.rand(d, k)
    A = orthogonal_(A, random_state)

    # A = scipy.linalg.orth(A)
    np.abs(A, out=A)
    A[A < 1e-6] = 1e-6

    B = np.random.rand(k, r)
    B = orthogonal_(B, random_state)

    Y, _ = eig_lastk(Ls, r)

    M = np.eye(k)  # init M
    # Lm=np.eye(k)
    F = np.zeros((n, k))  # init F

    max_iter = 30
    obj = np.zeros(max_iter)
    obj_a = np.zeros(max_iter)
    for iter_ in range(max_iter):
        # update S
        XAB = X @ A @ B
        dist_XAB = pairwise_distances(XAB) ** 2
        dist_y = pairwise_distances(Y) ** 2
        dist_f = - 2 * eta * F @ M @ F.T + lambda_f * dist_y
        S = local_structure_learning(kn, lambda_s + eta, dist_XAB, dist_f, 1)
        Ls = np.diag(S.sum(0)) - S

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
        XLX = X.T @ Ls @ X
        A_down = XLX @ A @ B @ B.T + alpha * XX @ A @ Lm + lambda_a * A @ A.T @ A
        temp = np.divide(lambda_a * A, A_down)
        A = A * np.array(temp)

        temp = np.diag(np.sqrt(np.diag(1 / (np.dot(A.transpose(), A) + 1e-16))))
        A = np.dot(A, temp)

        # update B
        B, _ = eig_lastk(A.T @ XLX @ A, r)

        # update Y
        Y_old = Y
        Y, e_val = eig_lastk(Ls, r)

        fn1 = np.sum(e_val[:r])
        fn2 = np.sum(e_val[:r + 1])
        # print(fn1,fn2)
        if fn1 > 10e-10:
            lambda_f = 2 * lambda_f
        elif fn2 < 10e-10:
            lambda_f = lambda_f / 2
            Y = Y_old
        else:
            break

    return A, S, M

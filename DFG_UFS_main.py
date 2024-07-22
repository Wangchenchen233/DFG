# -*- coding: utf-8 -*-
# @Time    : 3/27/2024 7:54 PM
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : DFG_UFS_main.py
import os.path
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from DFG_UFS import DFG_UFS
from utility.data_load import dataset_pro, construct_label_matrix
from utility.subfunc import estimateReg
from utility.unsupervised_evaluation import cluster_evaluation2
from utility.unsupervised_evaluation import nmi_to_excel

if __name__ == '__main__':
    # load data
    # X, y, Classes = dataset_pro('lung_discrete', 'scale')
    Data_names = ['lung_discrete']

    file_path = './results_DFG_UFS/'  # 文件路径和名称
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    for data_name in Data_names:
        print("data_name", data_name)
        X, y, n_Classes = dataset_pro(data_name, 'scale')

        Y = construct_label_matrix(y)

        Dist_x = pairwise_distances(X) ** 2
        Local_reg, S = estimateReg(Dist_x, 10)
        S = (S + S.T) / 2

        n_run = np.arange(1, 11, 1)
        Fea_nums = np.arange(1, 11, 1) * n_Classes
        # Fea_nums=10
        for fea_num in Fea_nums:
            nmi_results_random = []
            acc_results_random = []
            ne_results_random = []
            for i_run in n_run:
                paras = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                grid_search = [(alpha, eta) for alpha in paras for eta in paras]
                nmi_result = np.zeros((len(grid_search), 2))
                acc_result = np.zeros((len(grid_search), 2))
                ne_result = np.zeros((len(grid_search), 2))
                kk = 0
                pbar = tqdm(grid_search, desc=f'{data_name}, Fea_num={fea_num}, Run={i_run}')

                for i_para, para in enumerate(pbar):
                    alpha, eta = para

                    A_new, S_new, M_new = DFG_UFS(X, S, alpha, eta, lambda_s=Local_reg, k=fea_num, r=n_Classes, kn=10,
                                                  random_state=None)

                    idx = np.argsort(A_new.sum(1), 0)[::-1]
                    nmi_para_temp, acc_para_temp, ne_para_temp = cluster_evaluation2(X, y, n_Classes, idx, 20, [fea_num])

                    nmi_result[kk, :] = nmi_para_temp
                    acc_result[kk, :] = acc_para_temp
                    ne_result[kk, :] = ne_para_temp
                    kk += 1
                print(np.max(nmi_result[:,0]),np.max(acc_result[:, 0]))
                nmi_results_random.append(nmi_result)
                acc_results_random.append(acc_result)
                ne_results_random.append(ne_result)

            nmi_to_excel(file_path+'-nmi-'+str(fea_num), data_name, nmi_results_random, n_run, grid_search)
            nmi_to_excel(file_path+'-acc-'+str(fea_num), data_name, acc_results_random, n_run, grid_search)
            nmi_to_excel(file_path+'-ne-'+str(fea_num), data_name, ne_results_random, n_run, grid_search)

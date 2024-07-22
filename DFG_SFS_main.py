# -*- coding: utf-8 -*-
# @Time    : 3/27/2024 4:33 PM
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : DFG_SFS_main.py

import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utility.data_load import dataset_pro, construct_label_matrix
from DFG_SFS import rank_consistency_feature
from utility.construct_W import construct_W

def classification_verify(X_sub_train, X_sub_test, y_train, y_test):
    Cs = [0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1]

    accuracies = {}
    f1_scores = {}

    for C in Cs:
        for gamma in gammas:
            svm_model = SVC(C=C, kernel='rbf', gamma=gamma)
            svm_model.fit(X_sub_train, y_train)
            y_pred = svm_model.predict(X_sub_test)
            accuracy_key = f"SVM_C={C}_gamma={gamma}"
            accuracies[accuracy_key] = accuracy_score(y_test, y_pred)
            f1_scores[accuracy_key] = f1_score(y_test, y_pred, average='weighted')

    for k in [1, 3, 5, 7, 9]:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_sub_train, y_train)
        knn_pred = knn_model.predict(X_sub_test)
        accuracy_key = f"KNN_K={k}"
        accuracies[accuracy_key] = accuracy_score(y_test, knn_pred)
        f1_scores[accuracy_key] = f1_score(y_test, knn_pred, average='weighted')

    rf_model = RandomForestClassifier()
    rf_model.fit(X_sub_train, y_train)
    rf_pred = rf_model.predict(X_sub_test)
    accuracies["Random Forest"] = accuracy_score(y_test, rf_pred)
    f1_scores["Random Forest"] = f1_score(y_test, rf_pred, average='weighted')

    nb_model = GaussianNB()
    nb_model.fit(X_sub_train, y_train)
    y_pred = nb_model.predict(X_sub_test)
    accuracies["Naive Bayes"] = accuracy_score(y_test, y_pred)
    f1_scores["Naive Bayes"] = f1_score(y_test, y_pred, average='weighted')
    return accuracies, f1_scores


if __name__ == '__main__':
    # load data
    # X, y, Classes = dataset_pro('lung_discrete', 'scale')
    Data_names = ['lung_discrete']

    file_path = './results_DFG_SFS/'
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    for dname in Data_names:
        X, y, n_Classes = dataset_pro(dname, 'scale')
        Y = construct_label_matrix(y)

        Fea_nums = np.arange(1, 11, 1) * n_Classes

        n_run = 10  # repeat times
        n_split = 10  # Number of data set partitions
        n_model = 16  # Number of classifiers

        model_names = []

        paras = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        grid_search = [eta for eta in paras]

        # 遍历每个特征
        results_fea_all_acc = []
        results_fea_all_f1 = []
        for i_fea, fea_num in enumerate(Fea_nums):
            results_repeat_acc = np.zeros((n_run, n_model))
            results_repeat_f1 = np.zeros((n_run, n_model))
            for i_run in range(n_run):
                results_paras_acc = np.zeros((len(grid_search), n_model))
                results_paras_f1 = np.zeros((len(grid_search), n_model))
                pbar = tqdm(grid_search, desc=f'{dname}, Fea_num={fea_num}, Run={i_run + 1}')
                for i_para, eta in enumerate(pbar):

                    accuracies_para = np.zeros((n_split, n_model))
                    f1_score_para = np.zeros((n_split, n_model))
                    for i_split in range(n_split):

                        X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(X, y, Y, test_size=0.2,
                                                                                             random_state=i_split)

                        kwargs = {'neighbor_mode': 'supervised', 'fisher_score': True, 'metric': 'euclidean', 't': 1,
                                  'y': y_train, 'k': 10}
                        S = construct_W(X_train, **kwargs)
                        S = np.array(S.todense())
                        S = (S + S.T) / 2

                        A_new, M_new = rank_consistency_feature(X_train, S, alpha=1, eta=eta, k=fea_num, r=n_Classes,
                                                                random_state=None)
                        idx = np.argsort(A_new.sum(1), 0)[::-1]

                        X_sub = X_train[:, idx[0:fea_num]]
                        X_sub_train = X_train[:, idx[0:fea_num]]
                        X_sub_test = X_test[:, idx[0:fea_num]]

                        result_accs, result_f1s = classification_verify(X_sub_train, X_sub_test, y_train, y_test)
                        for i, (key, value) in enumerate(result_accs.items()):
                            accuracies_para[i_split, i] = value
                        for i, (key, value) in enumerate(result_f1s.items()):
                            f1_score_para[i_split, i] = value
                        if not model_names:
                            model_names = list(result_accs.keys())

                    results_paras_acc[i_para, :] = np.mean(accuracies_para, axis=0)
                    results_paras_f1[i_para, :] = np.mean(f1_score_para, axis=0)

                results_repeat_acc[i_run, :] = np.max(results_paras_acc, 0)
                results_repeat_f1[i_run, :] = np.max(results_paras_f1, 0)
                print(np.max(results_repeat_acc[i_run, :]),np.max(results_repeat_f1[i_run, :]))
            results_fea_all_acc.append(results_repeat_acc)
            results_fea_all_f1.append(results_repeat_f1)


        writer = pd.ExcelWriter(file_path + dname + "_acc.xlsx")
        for i, results_paras in enumerate(results_fea_all_acc):
            results_paras_df = pd.DataFrame(results_paras, columns=model_names)
            results_paras_df.to_excel(writer, sheet_name=f'Fea_num={Fea_nums[i]}', index=False)
        writer.save()
        writer = pd.ExcelWriter(file_path + dname + "_f1.xlsx")
        for i, results_paras in enumerate(results_fea_all_f1):
            results_paras_df = pd.DataFrame(results_paras, columns=model_names)
            results_paras_df.to_excel(writer, sheet_name=f'Fea_num={Fea_nums[i]}', index=False)
        writer.save()


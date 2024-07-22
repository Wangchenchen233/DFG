import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import scipy.io as scio
import sklearn
import pandas as pd


def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    g = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            g[i, j] = np.count_nonzero(ss & tt)

    aa = linear_assignment(-g)
    aa = np.asarray(aa)
    aa = np.transpose(aa)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[aa[i][1]]] = label1[aa[i][0]]
    return new_l2.astype(int)


def evaluation(x_selected, n_clusters, y):
    """
    This function calculates ARI, ACC and NMI of clustering results

    Input
    -----
    X_selected: {numpy array}, shape (n_samples, n_selected_features}
            input data on the selected features
    n_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels

    Output
    ------
    nmi: {float}
        Normalized Mutual Information
    acc: {float}
        Accuracy

        k_means = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001, precompute_distances=True, verbose=0,
                     random_state=None, copy_x=True, n_jobs=1)
    """


    k_means = KMeans(n_clusters=n_clusters, tol=0.0001)
    k_means.fit(x_selected)
    y_predict = k_means.labels_ + 1

    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict)

    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)
    # acc=0
    return nmi, acc

def evaluation2(x_selected, n_clusters, y):
    k_means = KMeans(n_clusters=n_clusters, tol=0.0001)
    k_means.fit(x_selected)
    y_predict = k_means.labels_ + 1

    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict)

    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)
    ne=normalized_entropy(n_clusters, y, y_predict)
    # acc=0
    return nmi, acc, ne

def normalized_entropy(c, y, y_pre):
    # c为类别个数
    # 将 y_pre 和 y 中的所有唯一类别进行合并
    unique_labels = np.union1d(np.unique(y_pre), np.unique(y))
    N_cluster = [np.sum(y_pre == label) for label in unique_labels]
    aa = []
    for i in range(c):
        N = np.sum(N_cluster)
        Ni = N_cluster[i] + np.finfo(float).eps
        a = Ni / N * np.log(Ni / N)
        aa.append(a)

    n_entropy = -1 / (np.log(c)) * np.sum(aa)  # 簇分布的熵; (0,1)
    return n_entropy



def cluster_evaluation2(data, label, classes, idx, cluster_times, feature_nums):
    """
    different feature num with fixed cluster times
    :param classes:
    :param label:
    :param data: n d
    :param idx: feature weight idx
    :param cluster_times: 20
    :param feature_nums: [50, 100, 150, 200, 250, 300]
    """
    nmi_fs_cluster_times = []
    acc_fs_cluster_times = []
    ne_fs_cluster_times=[]

    for feature_num in feature_nums:

        x_selected = data[:, idx[:feature_num]]
        # F = myspectral(x_selected, classes, 'RatioCut', 5)

        nmi_cluster_times = []
        acc_cluster_times = []
        ne_cluster_times = []
        for i in range(cluster_times):
            nmi, acc,ne = evaluation2(x_selected, classes, label)
            nmi_cluster_times.append(nmi)
            acc_cluster_times.append(acc)
            ne_cluster_times.append(ne)
        # print('feature num:', feature_num)
        # print(" NMI: {}+/-{}".format(np.mean(nmi_cluster_times, 0), np.std(nmi_cluster_times, 0)))
        # print(" ACC: {}+/-{}".format(np.mean(acc_cluster_times, 0), np.std(acc_cluster_times, 0)))

        # if relationship:
        #     from utility.relationship import calculate_erd
        #     rate = calculate_erd(x_selected, label, corr)
        #     print("rate", rate)
        # else:
        #     rate = 0
        nmi_fs_cluster_times.append([np.mean(nmi_cluster_times, 0), np.std(nmi_cluster_times, 0)])
        acc_fs_cluster_times.append([np.mean(acc_cluster_times, 0), np.std(acc_cluster_times, 0)])
        ne_fs_cluster_times.append([np.mean(ne_cluster_times, 0), np.std(ne_cluster_times, 0)])
    # print('cluster evaluation done')
    return np.array(nmi_fs_cluster_times), np.array(acc_fs_cluster_times), np.array(ne_fs_cluster_times)


def nmi_to_excel(filepath, data_name,acc_results_random,random_seeds,grid_search):
    import pandas as pd

    # 创建空的 DataFrame
    paras_str = [str(para) for para in grid_search]
    df = pd.DataFrame()

    # 遍历每个结果数组，将其添加到 DataFrame 中
    for i, result in enumerate(acc_results_random):
        result_df = pd.DataFrame(result, columns=[f"{random_seeds[i]} - mean", f"var"])
        df = pd.concat([df, result_df], axis=1)
    df.index = paras_str

    excel_file = filepath+data_name+'.xlsx'
    df.to_excel(excel_file)

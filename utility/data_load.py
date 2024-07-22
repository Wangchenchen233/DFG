import numpy as np
import scipy.io as scio
import sklearn
from sklearn import preprocessing


def dataset_pro(data_name, methods):
    """
    数据处理和生成
    :param data_name: .m file, 'X': n,d; 'Y': n,1
    :return: 'X': n,d; 'Y': n,1
    """
    data_old = scio.loadmat('./dataset/' + data_name + '.mat')
    label = data_old["Y"].astype('int')  # n 1
    unique_label = np.unique(label)
    classes = unique_label.shape[0]
    if methods == 'minmax':
        minmaxscaler = sklearn.preprocessing.MinMaxScaler()
        x = minmaxscaler.fit_transform(data_old["X"])
    elif methods == 'scale':
        x = preprocessing.scale(data_old["X"])  # n d
    else:
        x = np.array(data_old["X"])
    return x, label.reshape((label.shape[0],)), classes



def construct_label_matrix(label):
    """
    This function converts a 1d numpy array to a 2d array, for each instance, the class label is 1 or 0

    Input:
    -----
    label: {numpy array}, shape(n_samples,)

    Output:
    ------
    label_matrix: {numpy array}, shape(n_samples, n_classes)
    """

    n_samples = label.shape[0]
    unique_label = np.unique(label)
    n_classes = unique_label.shape[0]
    label_matrix = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        label_matrix[label == unique_label[i], i] = 1

    return label_matrix.astype(int)

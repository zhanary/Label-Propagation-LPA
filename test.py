# coding: utf8

import time
import math
import numpy as np
from lpa import labelPropagation
import matplotlib.pyplot as plt
import dataTool
import random
from parallel import run_label_propagation_sparse

plt.switch_backend('agg')  #避免plt在ssh情况下报错

# main function
if __name__ == "__main__":

    ########option###########
    unlabel_ratio = 0.8
    iter_num = 2000
    model = 0  #0为串行，1为并行
    knn_num = 10
    file_name = 'D31'
    acc = 1e-1
    ##########################

    mat_label_list = []
    mat_unlabel_list = []
    labels = []
    groundtruth = []
    labels_ids = set()
    labels_id = []


    all_data, all_label = dataTool.dataTool.file2matrix('data/D31.txt')
    for it in all_data:
        if random.random() < unlabel_ratio:
            mat_unlabel_list.append([float(it[0]), float(it[1])])
            groundtruth.append(int(it[2])-1)
        else:
            mat_label_list.append([float(it[0]), float(it[1])])
            labels.append(int(it[2])-1)

    labels_ids = set(groundtruth)
    for i in labels_ids:
        labels_id.append(i)

    Mat_Label = np.array(mat_label_list)
    Mat_Unlabel = np.array(mat_unlabel_list)

    ## Notice: when use 'rbf' as our kernel, the choice of hyper parameter 'sigma' is very import! It should be
    ## chose according to your dataset, specific the distance of two data points. I think it should ensure that
    ## each point has about 10 knn or w_i,j is large enough. It also influence the speed of converge. So, may be
    ## 'knn' kernel is better!
    # unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.2)
    start = time.process_time()
    if model == 0:
        unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, knn_num_neighbors=knn_num,
                                           max_iter=iter_num, tol=1e-1)
    else:
        unlabel_data_labels = run_label_propagation_sparse(Mat_Label, labels, labels_id, Mat_Unlabel, groundtruth, knn_num_neighbors=knn_num, max_iter=iter_num, tol=1e-1)


    end = time.process_time()

    dataTool.dataTool.draw_nodes(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels, file_name)

    print('time is', end - start)

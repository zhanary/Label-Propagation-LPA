# coding: utf8

#######################
# Author: ZouHang
# StudentID: 1809853P-II20-0032
# Date: 2019-04-19
#######################

import os, sys, time
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, eye
import operator
import pickle as pickle
import mpi4py.MPI as MPI

#
#   Global variables for MPI
#

# instance for invoking MPI related functions
comm = MPI.COMM_WORLD
# the node rank in the whole community
comm_rank = comm.Get_rank()
# the size of the whole community, i.e., the total number of working nodes in the MPI cluster
comm_size = comm.Get_size()



# return k neighbors index
def navie_knn(dataSet, query, k):
    numSamples = dataSet.shape[0]

    ## step 1: calculate Euclidean distance
    diff = np.tile(query, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row

    ## step 2: sort the distance
    sortedDistIndices = np.argsort(squaredDist)
    if k > len(sortedDistIndices):
        k = len(sortedDistIndices)
    return sortedDistIndices[0:k]


# build a big graph (normalized weight matrix)
# sparse U x (U + L) matrix
def buildSubGraph(Mat_Label, Mat_Unlabel, knn_num_neighbors):
    num_unlabel_samples = Mat_Unlabel.shape[0]
    data = [];
    indices = [];
    indptr = [0]
    Mat_all = np.vstack((Mat_Label, Mat_Unlabel))
    values = np.ones(knn_num_neighbors, np.float32) / knn_num_neighbors
    for i in range(num_unlabel_samples):
        k_neighbors = navie_knn(Mat_all, Mat_Unlabel[i, :], knn_num_neighbors)
        indptr.append(np.int32(indptr[-1]) + knn_num_neighbors)
        indices.extend(k_neighbors)
        data.append(values)
    return csr_matrix((np.hstack(data), indices, indptr))


# build a big graph (normalized weight matrix)
# sparse U x (U + L) matrix
def buildSubGraph_MPI(Mat_Label, Mat_Unlabel, knn_num_neighbors):
    num_unlabel_samples = Mat_Unlabel.shape[0]
    local_data = [];
    local_indices = [];
    local_indptr = [0]
    Mat_all = np.vstack((Mat_Label, Mat_Unlabel))
    values = np.ones(knn_num_neighbors, np.float32) / knn_num_neighbors
    sample_offset = np.linspace(0, num_unlabel_samples, comm_size + 1).astype('int')
    for i in range(sample_offset[comm_rank], sample_offset[comm_rank + 1]):
        k_neighbors = navie_knn(Mat_all, Mat_Unlabel[i, :], knn_num_neighbors)
        local_indptr.append(np.int32(local_indptr[-1]) + knn_num_neighbors)
        local_indices.extend(k_neighbors)
        local_data.append(values)
    data = np.hstack(comm.allgather(local_data))
    indices = np.hstack(comm.allgather(local_indices))
    indptr_tmp = comm.allgather(local_indptr)
    indptr = []
    for i in range(len(indptr_tmp)):
        if i == 0:
            indptr.extend(indptr_tmp[i])
        else:
            last_indptr = indptr[-1]
            del (indptr[-1])
            indptr.extend(indptr_tmp[i] + last_indptr)
    return csr_matrix((np.hstack(data), indices, indptr), dtype=np.float32)


# label propagation
def run_label_propagation_sparse(Mat_Label, labels, labels_id, Mat_Unlabel, unlabel_data_id, knn_num_neighbors=20, max_iter=100, tol=1e-3, test_per_iter=1):
    # load data and graph
    print('Processor %d/%d loading graph file...' % (comm_rank, comm_size))
    # Mat_Label, labels, Mat_Unlabel, groundtruth = loadFourBandData()
    # Mat_Label, labels, labels_id, Mat_Unlabel, unlabel_data_id = load_MNIST()
    if comm_size > len(labels_id):
        raise ValueError("Sorry, the processors must be less than the number of classes")
    # affinity_matrix = buildSubGraph(Mat_Label, Mat_Unlabel, knn_num_neighbors)
    affinity_matrix = buildSubGraph_MPI(Mat_Label, Mat_Unlabel, knn_num_neighbors)

    # get some parameters
    num_classes = len(labels_id)
    num_label_samples = len(labels)
    num_unlabel_samples = Mat_Unlabel.shape[0]

    affinity_matrix_UL = affinity_matrix[:, 0:num_label_samples]
    affinity_matrix_UU = affinity_matrix[:, num_label_samples:num_label_samples + num_unlabel_samples]

    if comm_rank == 0:
        print('Have %d labeled images, %d unlabeled images and %d classes' % (num_label_samples, num_unlabel_samples, num_classes))

    # divide label_function_U and label_function_L to all processors
    class_offset = np.linspace(0, num_classes, comm_size + 1).astype('int')

    # initialize local label_function_U
    local_start_class = class_offset[comm_rank]
    local_num_classes = class_offset[comm_rank + 1] - local_start_class
    local_label_function_U = eye(num_unlabel_samples, local_num_classes, 0, np.float32, format='csr')

    # initialize local label_function_L
    local_label_function_L = lil_matrix((num_label_samples, local_num_classes), dtype=np.float32)
    for i in range(num_label_samples):
        class_off = int(labels[i]) - local_start_class
        if class_off >= 0 and class_off < local_num_classes:
            local_label_function_L[i, class_off] = 1.0
    local_label_function_L = local_label_function_L.tocsr()
    local_label_info = affinity_matrix_UL.dot(local_label_function_L)
    print('Processor %d/%d has to process %d classes...' % (comm_rank, comm_size, local_label_function_L.shape[1]))

    # start to propagation
    iter = 1;
    changed = 100.0;
    evaluation(num_unlabel_samples, local_start_class, local_label_function_U, unlabel_data_id, labels_id)
    while True:
        pre_label_function = local_label_function_U.copy()

        # propagation
        local_label_function_U = affinity_matrix_UU.dot(local_label_function_U) + local_label_info

        # check converge
        local_changed = abs(pre_label_function - local_label_function_U).sum()
        changed = comm.reduce(local_changed, root=0, op=MPI.SUM)
        status = 'RUN'
        test = False
        if comm_rank == 0:
            if iter % 1 == 0:
                norm_changed = changed / (num_unlabel_samples * num_classes)
                print('---> Iteration %d/%d, changed: %f' % (iter, max_iter, changed))
            if iter >= max_iter or changed < tol:
                status = 'STOP'
                print('************** Iteration over! ****************')
            if iter % test_per_iter == 0:
                test = True
            iter += 1
        test = comm.bcast(test if comm_rank == 0 else None, root=0)
        status = comm.bcast(status if comm_rank == 0 else None, root=0)
        if status == 'STOP':
            break
        if test == True:
            evaluation(num_unlabel_samples, local_start_class, local_label_function_U, unlabel_data_id, labels_id)
    evaluation(num_unlabel_samples, local_start_class, local_label_function_U, unlabel_data_id, labels_id)
    return unlabel_data_id

def evaluation(num_unlabel_samples, local_start_class, local_label_function_U, unlabel_data_id, labels_id):
    # get local label with max score
    if comm_rank == 0:
        print('Start to combine local result...')
    local_max_score = np.zeros((num_unlabel_samples, 1), np.float32)
    local_max_label = np.zeros((num_unlabel_samples, 1), np.int32)
    for i in range(num_unlabel_samples):
        local_max_label[i, 0] = np.argmax(local_label_function_U.getrow(i).todense())
        local_max_score[i, 0] = local_label_function_U[i, local_max_label[i, 0]]
        local_max_label[i, 0] += local_start_class

    # gather the results from all the processors
    if comm_rank == 0:
        print
        "Start to gather results from all processors"
    all_max_label = np.hstack(comm.allgather(local_max_label))
    all_max_score = np.hstack(comm.allgather(local_max_score))

    # get terminate label of unlabeled data
    if comm_rank == 0:
        print
        "Start to analysis the results..."
        right_predict_count = 0
        for i in range(num_unlabel_samples):
            if i % 1000 == 0:
                print('***', all_max_score[i])
            max_idx = np.argmax(all_max_score[i])
            max_label = all_max_label[i, max_idx]
            if int(unlabel_data_id[i]) == int(labels_id[max_label]):
                right_predict_count += 1
        accuracy = float(right_predict_count) * 100.0 / num_unlabel_samples
        print('Have %d samples, accuracy: %.3f%%' % (num_unlabel_samples, accuracy))



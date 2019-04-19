# encoding=utf8

import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
import string
from numpy import *

# plt.switch_backend('agg')  #避免plt在ssh情况下报错


class dataTool:

    def __init__(self):
        pass

    #转换数据
    @staticmethod
    def file2matrix(filename):
        fr = open(filename)
        numberOfLines = len(fr.readlines()) # get the number of lines in the file
        returnMat = zeros((numberOfLines, 3)) # prepare matrix to return
        classLabelVector = []  # prepare labels return
        fr = open(filename)
        index = 0
        for line in fr.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector

    @staticmethod
    def loadData(filePath):
        f = open(filePath)
        vector_dict = {}
        edge_dict = {}
        for line in f.readlines():
            #print('line',line)
            lines = line.strip().split()
            #print('lines',lines)
            for i in range(2):
                #print(lines[2])
                if lines[i] not in vector_dict:
                    vector_dict[lines[i]] = float(lines[i])
                    edge_list = []
                    if len(lines) == 3:
                        edge_list.append(lines[1 - i] + ":" + lines[2])
                    else:
                        edge_list.append(lines[1 - i] + ":" + "1")
                    edge_dict[lines[i]] = edge_list
                else:
                    edge_list = edge_dict[lines[i]]
                    if len(lines) == 3:
                        edge_list.append(lines[1 - i] + ":" + lines[2])
                    else:
                        edge_list.append(lines[1 - i] + ":" + "1")
                    edge_dict[lines[i]] = edge_list

        return vector_dict, edge_dict

    @staticmethod
    def draw_nodes(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels, file_name):
        data_mat_list = []
        for i in range(len(labels)):
            data_mat_list.append([Mat_Label[i, 0], Mat_Label[i, 1], labels[i]])
        for i in range(len(unlabel_data_labels)):
            data_mat_list.append([Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], unlabel_data_labels[i]])
        data_mat = np.matrix(data_mat_list)
        plt.scatter(data_mat[:, 0].tolist(), data_mat[:, 1].tolist(), c=data_mat[:, 2].tolist(), alpha=0.3, edgecolors='white')

        plt.xlabel('%s - result'%file_name)
        plt.grid(True)
        #plt.legend()

        plt.show()
        plt.savefig('%s.jpg'%file_name)




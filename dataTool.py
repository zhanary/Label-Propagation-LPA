import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
from numpy import *

plt.switch_backend('agg')  #避免plt在ssh情况下报错

#转换数据
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

    return vector_dict, edge_dict, lines

if __name__ == '__main__':
    filePath = 'spiral.txt'

    ax1 = plt.subplot(111)

    vector, edge, lines = loadData(filePath)

    #print(vector)
    #print(edge)
    data_mat, label = file2matrix(filePath)
    print(data_mat[:, 2])

    plt.scatter(data_mat[:, 0], data_mat[:, 1], c=data_mat[:, 2], alpha=0.3, edgecolors='white')

    plt.xlabel('spiral')
    plt.grid(True)
    #plt.legend()

    plt.show()
    plt.savefig('spiral')




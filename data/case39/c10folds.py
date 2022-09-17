

import numpy as np
from scipy.io import loadmat

data = loadmat('case39.mat')
label = data['data']['label'][0][0]
# bus_matrix = data['data']['bus_matrix'][0][0]
# line = data['data']['line'][0][0]



label0 = np.argwhere(label == 0)
label1 = np.argwhere(label == 1)
np.random.shuffle(label0)
np.random.shuffle(label1)
# label1 = label1[:int(len(label1)/2.8)]
num_fold_sample0 = int(len(label0)/10)
num_fold_sample1 = int(len(label1)/10)
label0_endInd = num_fold_sample0 * 10
label1_endInd = num_fold_sample1 * 10

ind_train_matrix = np.zeros([10, 9*(num_fold_sample0 + num_fold_sample1)])
ind_test_matrix = np.zeros([10, num_fold_sample0 + num_fold_sample1])

np.random.shuffle(label0)
np.random.shuffle(label1)
ind_start0 = 0
ind_start1 = 0

for i in range(10):
    ind_end0 = ind_start0 + num_fold_sample0
    ind_end1 = ind_start1 + num_fold_sample1
    label0_test = label0[ind_start0:ind_end0, 0]
    label1_test = label1[ind_start1:ind_end1, 0]
    label_test = np.append(label0_test,label1_test)

    label0_train1 = label0[0:ind_start0, 0]
    label0_train2 = label0[ind_end0:label0_endInd, 0]
    label0_train = np.append(label0_train1,label0_train2)

    label1_train1 = label1[0:ind_start1, 0]
    label1_train2 = label1[ind_end1:label1_endInd, 0]
    label1_train = np.append(label1_train1,label1_train2)

    label_train = np.append(label0_train,label1_train)

    ind_train_matrix[i,:] = label_train
    ind_test_matrix[i,:] = label_test

    ind_start0 = ind_end0
    ind_start1 = ind_end1

ind_train_matrix = ind_train_matrix.astype(int)
ind_test_matrix = ind_test_matrix.astype(int)
np.save('ind_train_matrix.npy', ind_train_matrix)
np.save('ind_test_matrix.npy', ind_test_matrix)



# -*- coding: utf-8 -*-
"""
Created on Tue Apr 9 15:15:50 2020
@author: Patrick
"""

"""Implementation of Fast Orthogonal Search"""

#==============================================
#main.py
#==============================================

import numpy as np
import copy
from matplotlib import pyplot as plt
from nonlinear_data_generation import Nonlinear_Generation
from candidates_generation import CandidatePool_Generation
from FOS import FOS
from multiprocessing import Pool

#_________________________________Input Data___________________________________#
mu, sigma = 0, 1 # zero mean, 1.0 standard deviation
data_length = 3000
[train_length, val_length, test_length] = [1000, 1000, 1000]

x = np.random.normal(mu, sigma, data_length) # Random Gaussian Mean generation

y = np.zeros((data_length, 1))

p_array = np.asarray([0.3, 0.5, 0.7, 1.0]) # Different P values for noisy data
pred_list=[]
test_list=[]
free_list=[]

# for num in range(1,4):
# Three Difference Equations of Structure
for p_num in range(0, 4):
    # Nonlinear Data Generation
    # if noise-free
    # nonlinear_data = Nonlinear_Generation(mu, sigma, x, y, 0, num, False)
    # if noisy
    nonlinear_data = Nonlinear_Generation(mu, sigma, x, y, p_array[p_num], 2,True)
    noise_free_data =Nonlinear_Generation(mu, sigma, x, y, 0, 2, False)

    # y = copy.deepcopy(nonlinear_data)
    y = nonlinear_data

    # Obtain Training, Validation and Testing Data
    x_train = x[ :train_length]
    y_train = y[ :train_length]
    x_val   = x[train_length: train_length + val_length]
    y_val   = y[train_length: train_length + val_length]
    x_test  = x[train_length + val_length: ]
    y_test  = y[train_length + val_length: ]
    y_free  = noise_free_data[train_length + val_length: ]



    #_______________________________Training Phase____________________________#
    # 10 Different Combinations of K and L

    KL_Comb = [[10,10], [10, 7], [7,10], [7,7], [7,5], [5,7], [5,5], [5,3],
    [3,5], [3,3]]
    KL_Comb = np.asarray(KL_Comb)

    # Initialization
    a_list   = []
    MSE_list = []
    Idx_list = []
    P_list   = []


    for i in range(KL_Comb.shape[0]):

        K, L = KL_Comb[i]
        N0 = np.max([K,L])

        CandidatePool = CandidatePool_Generation(x_train, y_train, K, L)
        CandidatePool = np.asarray(CandidatePool)
        CandidatePool = CandidatePool.T

        a, MSE,Idx,P = FOS(CandidatePool, y_train, N0, True)
        a_list.append(a)
        MSE_list.append(MSE)
        Idx_list.append(Idx)
        P_list.append(P)



    #____________________________Validation Phase______________________________#
    # 10 Different Combinations of K and L

    MSE_val_list = []
    Idx_list = np.asarray(Idx_list)

    for i in range(KL_Comb.shape[0]):
        K, L = KL_Comb[i]
        N0 = np.max([K,L])

        CandidatePool = CandidatePool_Generation(x_val, y_val, K, L)
        CandidatePool = np.asarray(CandidatePool)
        CandidatePool = CandidatePool.T
        # Best Selected Candidates
        P_selected  = CandidatePool[:, Idx_list[i].astype(int)]
        Ones_Column = np.zeros((P_selected.shape[0], P_selected.shape[1]+1))
        Ones_Column[:,1:] = P_selected
        P_selected  = Ones_Column
        a_list[i] = np.asarray(a_list[i])
        y_pred = np.expand_dims(np.sum(P_selected* a_list[i], axis=1), axis=1)

        MSE_val = np.mean((y_val - y_pred)**2) / np.mean((y_val-np.mean(y_val))**2)*100
        MSE_val_list.append(MSE_val)

    MSE_val_list = np.asarray(MSE_val_list)
    Min_mse_index = np.argmin(MSE_val_list)

    #______________________________Testing Phase____________________________#

    K_test, L_test = KL_Comb[Min_mse_index] # Best Selected Model
    CandidatePool = CandidatePool_Generation(x_test, y_test, K_test, L_test)

    CandidatePool = np.asarray(CandidatePool)
    CandidatePool = CandidatePool.T

    # Best Selected Candidates
    P_selected  = CandidatePool[:, Idx_list[Min_mse_index].astype(int)]
    Ones_Column = np.zeros((P_selected.shape[0], P_selected.shape[1]+1))
    Ones_Column[:,1:] = P_selected
    P_selected  = Ones_Column

    a_list[Min_mse_index] = np.asarray(a_list[Min_mse_index])
    y_pred = np.expand_dims(np.sum(P_selected* a_list[Min_mse_index], axis=1), axis=1)
    # print(y_pred.shape)
    MSE_test = np.mean((y_free - y_pred)**2)/np.mean((y_free-np.mean(y_test))**2)*100

    pred_list.append(y_pred)
    test_list.append(y_test)
    free_list.append(y_free)


    print(MSE_val_list, "MSE_val_list",Min_mse_index, "Min_mse_index")
    print(MSE_test)


pred_list = np.asarray(pred_list)
test_list = np.asarray(test_list)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.plot(pred_list[0], 'r', label='z')
# ax1.plot(test_list[0], 'b', label='y')
# ax1.legend('zy')
# ax1.set_title('System No.1')
# ax2.plot(pred_list[1], 'r', label='z')
# ax2.plot(test_list[1], 'b', label='y')
# ax2.legend('zy')
# ax2.set_title('System No.2')
# ax3.plot(pred_list[2], 'r', label='z')
# ax3.plot(test_list[2], 'b', label='y')
# ax3.legend('zy')
# ax3.set_title('System No.3')
# plt.show()


# Draw for noisy output

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(pred_list[0], 'r', label="z[n]", linewidth=0.1)
axs[0, 0].plot(free_list[0], 'b', label="y[n]", linewidth=0.1)
axs[0, 0].legend("zy")
axs[0, 0].set_title('P=30')
axs[0, 1].plot(pred_list[1], 'r', label="z[n]", linewidth=0.1)
axs[0, 1].plot(free_list[1], 'b', label="y[n]", linewidth=0.1)
axs[0, 1].legend("zy")
axs[0, 1].set_title('P=50')
axs[1, 0].plot(pred_list[2], 'r', label="z[n]", linewidth=0.1)
axs[1, 0].plot(free_list[2], 'b', label="y[n]", linewidth=0.1)
axs[1, 0].legend("zy")
axs[1, 0].set_title('P=70')
axs[1, 1].plot(pred_list[3], 'r', label="z[n]", linewidth=0.1)
axs[1, 1].plot(free_list[3], 'b', label="y[n]", linewidth=0.1)
axs[1, 1].legend("zy")
axs[1, 1].set_title('P=100')

for ax in axs.flat:
    ax.set(xlabel='sample', ylabel='output')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()



#

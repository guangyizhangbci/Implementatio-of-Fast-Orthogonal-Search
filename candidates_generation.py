# -*- coding: utf-8 -*-
"""
Created on Tue Apr 9 15:15:50 2020
@author: Patrick
"""

"""Implementation of Fast Orthogonal Search"""

#==============================================
#candidates_generation.py
#==============================================

def CandidatePool_Generation(x_train, y_train, K, L):
    Candidates = []
    # x[n-l], l = 0,...,10 (11 Candidates)
    for l in range(0, L+1):
        zero_list = l * [0]
        data_list = x_train[:len(x_train)-l]
        xn_l = [*zero_list, *data_list]
        Candidates.append(xn_l)

    # y[n-k], l = 1,...,10 (10 Candidates)
    for k in range(1, K+1):
        zero_list = k * [0]
        data_list = y_train[:len(y_train)-k]
        yn_k = [*zero_list, *data_list]
        Candidates.append(yn_k)

    # x[n-l1]x[n-l2], l1 = 0,...,10
     # l2 = l1,...,10 (66 Candidates)
    for l1 in range(0, L+1):
        for l2 in range(l1, L+1):
            zero_list_l1 = l1 * [0]
            zero_list_l2 = l2 * [0]
            data_list_l1 = x_train[:len(x_train)-l1]
            data_list_l2 = x_train[:len(x_train)-l2]
            xn_l1 = [*zero_list_l1, *data_list_l1]
            xn_l2 = [*zero_list_l2, *data_list_l2]

            Candidates.append([i*j for i,j in zip(xn_l1,xn_l2)])

    # y[n-l1]y[n-l2], k1 = 1,...,10
    #  k2 = k1,...,10 (55 Candidates)
    for k1 in range(1, K+1):
        for k2 in range(k1, K+1):
            zero_list_k1 = k1 * [0]
            zero_list_k2 = k2 * [0]
            data_list_k1 = y_train[:len(y_train)-k1]
            data_list_k2 = y_train[:len(y_train)-k2]
            yn_k1 = [*zero_list_k1, *data_list_k1]
            yn_k2 = [*zero_list_k2, *data_list_k2]

            Candidates.append([i*j for i,j in zip(yn_k1,yn_k2)])

    # x[n-l]y[n-k], l = 1,...,10
    #  k = 1,...,10 (110 Candidates)
    for l in range(0, L+1):
        for k in range(1, K+1):
            zero_list_l = l * [0]
            zero_list_k = k * [0]
            data_list_l = x_train[:len(x_train)-l]
            data_list_k = y_train[:len(y_train)-k]
            xn_l = [*zero_list_l, *data_list_l]
            yn_k = [*zero_list_k, *data_list_k]

            Candidates.append([i*j for i,j in zip(xn_l,yn_k)])

    return Candidates

#

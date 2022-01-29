# -*- coding: utf-8 -*-
"""
Created on Tue Apr 9 15:15:50 2020
@author: Patrick
"""

"""Implementation of Fast Orthogonal Search"""

#==============================================
#nonlinear_data_generation.py
#==============================================

import numpy as np
from matplotlib import pyplot as plt
from FOS import FOS


def Nonlinear_Generation(mu, sigma, x, y, P, case_index, noise):
    # case 1:
    if case_index ==1:
        [a0, a1, a2, a3, a4, a5, a6] = [0.05, 0.4, 0.1, -0.2, -0.1, 0.33, 0.0]    # 2nd order

    # case 2:
    elif case_index ==2:
        [a0, a1, a2, a3, a4, a5, a6] = [0.01, 0.2, 0.3, -0.1, 0.05, 0.2, 0.0]  # 2nd order

    # case 3:
    else:
        [a0, a1, a2, a3, a4, a5, a6] = [0.1, 0.1, 0.5, -0.3, 0.22, -0.4, 0.1] # 3rd order

    for n in range(2, len(y)):
        y[n] =  a0 + a1*y[n-1]+ a2*x[n-1]+ a3*x[n]*x[n-2]+ a4*y[n-1]*y[n-2]
        +a5*x[n-2]*y[n-2]+ a6*x[n-1]*x[n-2]*y[n-2]
    yn = y + P*np.var(y)* np.expand_dims(np.random.normal(mu, sigma,len(x)), axis=1)

    if noise==True:
        y = yn
    else:
        y = y

    return y

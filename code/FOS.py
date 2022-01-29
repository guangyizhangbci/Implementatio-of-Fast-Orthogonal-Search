# -*- coding: utf-8 -*-
"""
Created on Tue Apr 9 15:15:50 2020
@author: Patrick
"""

"""Implementation of Fast Orthogonal Search"""

import numpy as np
from tqdm import tqdm

def FOS(CandidatePool, y, N0, Noise):
    N, M = CandidatePool.shape[0],CandidatePool.shape[1]
    P = np.ones((N, 1))  # Selected Candidates Pool
    D = np.zeros((M+1, M+1))
    C = np.zeros((M+1,))
    alpha = np.zeros((M+1, M))
    # Parameters Initialization
    D[0,0] = 1
    C[0] = np.mean(y)
    g = C[0]/D[0,0]
    Q = C[0]*C[0]/D[0,0]

    Idx = np.empty((0,1)) # Selected Index List with max Q
    for m in tqdm(range(1, M+1)):
        Qm  = np.empty((0,1))

        for i in range(1, M+1):
            pm = CandidatePool[:,i-1]
            pm = np.expand_dims(pm, axis=1)
            D[m,0] = np.mean(pm)

            if m == 1:
                alpha[1,0] = D[1,0]/D[0,0]
                SigmaD = alpha[1,0]* D[1,0]
                D[1,1] = np.mean(pm*pm)-SigmaD
                SigmaC = alpha[1,0]*C[0]

            else:
                SigmaC = 0
                for r in range(0, m):
                    alpha[m,r] = D[m,r]/D[r,r]

                    SigmaD = 0
                    for j in range(0, r+1):
                        SigmaD = SigmaD + alpha[r+1,j]* D[m,j]

                    if r < m-1:
                        P_update = np.expand_dims(P[:,r+1],axis=1)
                        D[m,r+1] = np.mean(pm*P_update)-SigmaD
                    else:
                        D[m,m] = np.mean(pm*pm)-SigmaD
                    SigmaC = SigmaC + alpha[m,r]*C[r]
            C[m] = np.mean(y*pm) - SigmaC
            if D[m,m] < np.exp(-30):
                Qt = 0
            else:
                Qt = C[m]*C[m]/D[m,m]
            Qm = np.append(Qm, Qt)

        [Qmax,Idxm] = [np.max(Qm), np.argmax(Qm)]

        # Stopping criterion value
        criterion_value = (4/(N-N0+1))* (np.mean(y*y)-np.sum(Q))

        if (Noise==True) and (Qmax < criterion_value): # Only while noisy data!
            print('Stop Seaching')
            break
        else: # Continue after model term pm[n] is selected
            pm  = CandidatePool[:,Idxm]
            pm  = np.expand_dims(pm, axis=1)
            Idx = np.append(Idx, Idxm)
            P   = np.append(P, pm, axis=1)

            D[m,0] = np.mean(pm)
            if m == 1:
                alpha[1,0] = D[1,0]/D[0,0]
                SigmaD = alpha[1, 0]* D[1,0]
                D[1,1] = np.mean(pm*pm)-SigmaD
                SigmaC = alpha[1,0]*C[0]
            else:
                SigmaC = 0
                for r in range(0, m): # r = 0,...m-1
                    alpha[m,r] = D[m,r]/D[r,r]
                    SigmaD = 0
                    for j in range(0, r+1): # for j = 0:r
                        SigmaD = SigmaD + alpha[r+1,j]* D[m,j] #
                        P_update = np.expand_dims(P[:,r+1],axis=1)
                        D[m,r+1] = np.mean(pm*P_update)-SigmaD
                    else:
                        D[m,m] = np.mean(pm*pm)-SigmaD

                    SigmaC = SigmaC + alpha[m,r]*C[r]
            C[m] = np.mean(y*pm) - SigmaC
            # Ensure D[m,m] exceeds a specified positive threshold level.
            if D[m,m] > np.exp(-30):
                Q = np.append(Q, C[m]*C[m]/D[m,m])
            else:
                continue
            g = np.append(g, C[m]/D[m,m])

   # Mean Squared Error Calculation
    MSE = np.mean(y*y)-np.sum(Q)

    # Coefficient a calculation
    a = np.empty((0,1))
    m = g.shape[0]-1

    for i in range(0,m):
        v = np.zeros((m+1,1))
        v[i] = 1
        for m in range(i+1, m+1):
            Vi = 0
            for r in range(i, m):
                Vi = Vi + alpha[m,r]*v[r]
            v[m] = -Vi
        Am = 0
        for j in range(i,m+1):
            Am = Am + g[j]*v[j]
        a = np.append(a, Am)
    a = np.append(a, g[m])

    return a, MSE, Idx, P



#

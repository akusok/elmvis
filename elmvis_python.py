# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
from time import time


def elmvis(Y, A, tol=1E-9, maxiter=100000, maxstall=10000, report=1000):
    N, d = Y.shape
    print "original error:", np.diag(Y.T.dot(A).dot(Y)).sum() / d

    AY = np.dot(A, Y)

    stall = 0
    iters = 0
    t = time()
    while (stall < maxstall) and (iters < maxiter):
        iters += 1
        stall += 1

        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        y1 = Y[i1, :].copy()
        y2 = Y[i2, :].copy()
        delta1 = y2 - y1
        delta2 = y1 - y2
        diff = A[i1,i1]*np.sum(delta1**2) + 2*np.sum(AY[i1]*delta1) +\
               A[i2,i2]*np.sum(delta2**2) + 2*np.sum(AY[i2]*delta2) + 2*A[i2,i1]*np.sum(delta1*delta2)

        if diff > -(tol * maxstall) / (iters + maxstall):
            stall = 0

            Y[i1] = y2
            Y[i2] = y1

            Ai = A[:, (i1, i2)]
            Deltas = np.vstack((delta2, delta1))
            AY -= np.dot(Ai, Deltas)

        if iters % report == 0:
            print "%d | %d | %.0f iters/min" % (iters, stall, report*60.0/(time()-t))
            t = time()

    bests = np.diag(Y.T.dot(A).dot(Y)).sum() / d
    return Y, bests




def permute(Y, s, A, i1, i2):
    snew = s.copy()  # python gives handlers, not actual matrices
    Ynew = Y.copy()
    y1 = Y[i1, :]
    y2 = Y[i2, :]
    delta1 = y2 - y1
    delta2 = y1 - y2

    snew += A[i1, i1]*delta1**2 + 2*np.dot(A[i1, :], Ynew)*delta1
    Ynew[i1, :] = y2

    snew += A[i2, i2]*delta2**2 + 2*np.dot(A[i2, :], Ynew)*delta2
    Ynew[i2, :] = y1

    return snew, Ynew



def elmvis_original(Y, A, tol=0, maxiter=100000, maxstall=10000, report=1000):
    N, d = Y.shape
    s = np.diag(Y.T.dot(A).dot(Y)).reshape(d)
    bests = s.sum()

    stall = 0
    iters = 0
    while (stall < maxstall) and (iters < maxiter):
        iters += 1
        stall += 1

        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        snew, Ynew = permute(Y, s, A, i1, i2)
        swap = snew.sum()

        if swap + tol > bests:
            bests = swap
            Y = Ynew
            s = snew
            stall = 0

        if iters % report == 0:
            print iters, stall

    return Y, bests












# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
import numexpr as ne
from time import time

#from pycuda import autoinit, gpuarray
#from skcuda import linalg


def scores(X, Y, A):
    """Scores, higher the better.
    """
    N, d = X.shape
    d = Y.shape[1]
    scores = np.zeros((N, 1))
    for i in xrange(N):
        p = X[i, :]
        yy = Y[p, :]
        scores[i] = np.trace(yy.T.dot(A).dot(yy)) / d
    return scores


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


def bench(Y, A, maxiter, nthreads=1):
    i1 = 1
    i2 = 2
    diff = 0
    t = time()
    for _ in xrange(maxiter):
        y1 = Y[i1, :]
        y2 = Y[i2, :]
        delta1 = y2 - y1
        delta2 = y1 - y2
        diff += np.sum(A[i1, i1]*delta1**2 + 2*np.dot(A[i1, :], Y)*delta1)
        diff += np.sum(A[i2, i2]*delta2**2 + 2*(np.dot(A[i2, :], Y) + A[i2, i1]*delta1)*delta2)
    t = time()-t
    return t, diff


#@profile
def opt(Y, A, tol=1E-9, maxiter=100000, maxstall=10000, report=1000, nthreads=1):
    N, d = Y.shape
    s = np.diag(Y.T.dot(A).dot(Y)).reshape(d).copy()
    bests = s.sum() / d
    AY = np.dot(A, Y)    

#    linalg.init()    
#    devA = gpuarray.to_gpu(A)
#    devY = gpuarray.to_gpu(Y)
#    devAY = linalg.dot(devA, devY)
#    devAi = gpuarray.empty((N, 2), dtype=np.float64)
#    devYi = gpuarray.empty((2, d), dtype=np.float64)

    stall = 0
    iters = 0
    ips = 0

    t = time()
    while (stall < maxstall) and (iters < maxiter):
        iters += 1
        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        y1 = Y[i1, :].copy()
        y2 = Y[i2, :].copy()
        delta1 = y2 - y1
        delta2 = y1 - y2

        diff = np.sum(A[i1, i1]*delta1**2 + 2*AY[i1]*delta1)
        diff += np.sum(A[i2, i2]*delta2**2 + 2*(AY[i2] + A[i2, i1]*delta1)*delta2)

        stall += 1
        if diff > -(tol * maxstall) / (iters + maxstall):
            s += A[i1, i1]*delta1**2 + 2*AY[i1]*delta1
            s += A[i2, i2]*delta2**2 + 2*(AY[i2] + A[i2, i1]*delta1)*delta2
            Y[i1] = y2
            Y[i2] = y1

#            devAi[:, 0] = devA[:, i1]
#            devAi[:, 1] = devA[:, i2]
#            devYi[0, :] = devY[i1, :] - devY[i2, :]
#            devYi[1, :] = devY[i2, :] - devY[i1, :]
#            linalg.add_dot(devAi, devYi, devAY, alpha=-1)
#            devY[i1, :] = y2
#            devY[i2, :] = y1
#            devAY.get(ary=AY)

            Ai = A.take((i1, i2), axis=1)
            Yi = np.vstack((delta2, delta1))
            AY -= np.dot(Ai, Yi)
            assert np.allclose(AY, AY2)

            bests = s.sum() / d
            stall = 0



        if iters % report == 0:
            ips = report*1.0/(time()-t)
            print "%d | %d | %.0f iters/min" % (iters, stall, ips*60)
            t = time()
    
    return Y, bests, ips


def opt_original(Y, A, maxiter=100000, maxstall=10000, report=1000):

    N, d = Y.shape
    s = np.diag(Y.T.dot(A).dot(Y)).reshape(d)
    bests = s.sum()

    stall = 0
    iters = 0
    while (stall < maxstall) and (iters < maxiter):
        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        snew, Ynew = permute(Y, s, A, i1, i2)

        swap = snew.sum()
        if swap > bests:
            bests = swap
            Y = Ynew
            s = snew
            stall = 0

        if iters % report == 0:
            print iters, stall
        iters += 1
        stall += 1

    return Y, bests












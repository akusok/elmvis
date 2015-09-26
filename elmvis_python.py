# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
import numexpr as ne
from time import time


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


def opt(Y, A, maxiter=100000, maxstall=10000, report=1000):
    N, d = Y.shape
    s = np.diag(Y.T.dot(A).dot(Y)).reshape(d)
    bests = s.sum() / d

    stall = 0
    iters = 0
    t = time()
    while (stall < maxstall) and (iters < maxiter):
        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        y1 = Y[i1, :]
        y2 = Y[i2, :]
        delta1 = y2 - y1
        delta2 = y1 - y2

        diff = np.sum(A[i1, i1]*delta1**2 + 2*np.dot(A[i1, :], Y)*delta1)
        diff += np.sum(A[i2, i2]*delta2**2 + 2*(np.dot(A[i2, :], Y) + A[i2, i1]*delta1)*delta2)

        if diff > -maxstall**0.5 / (maxstall + iters):
            s, Y = permute(Y, s, A, i1, i2)
            bests = s.sum() / d
            stall = 0

        stall += 1
        iters += 1        
        if iters % report == 0:
            print "%d | %d | %.0f iters/min" % (iters, stall, report*60.0/(time()-t))
            print bests, maxstall**0.5 / (maxstall + iters)
            t = time()


    return Y, bests


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












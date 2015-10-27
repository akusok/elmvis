# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
from time import time


def elmvis(X, A, tol=1E-9, cossim=None,
           maxiter=1000000,
           maxstall=1000,
           maxupdate=1000000,
           maxtime=24*60*60,
           report=1000,
           silent=False):
    """ELMVIS+ function.
    """

    N, d = X.shape
    if cossim is None:
        cossim = np.trace(X.T.dot(A).dot(X)) / d
    if not silent: print "original similarity: ", cossim
    tol = tol*cossim

    AX = np.dot(A, X)

    stall = 0
    iters = 0
    updates = 0
    t0 = tlast = time()

    # track a list of last "nstalls" updates, get their mean
    nstalls = 100
    stalls = np.ones((nstalls,))
    istall = 0

    while (iters < maxiter) and (stall < maxstall*10):
        iters += 1
        stall += 1

        # get two different random numbers
        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        x1 = X[i1, :].copy()
        x2 = X[i2, :].copy()
        delta1 = x2 - x1
        delta2 = x1 - x2
        diff = A[i1,i1]*np.sum(delta1**2) + 2*np.sum(AX[i1]*delta1) +\
               A[i2,i2]*np.sum(delta2**2) + 2*np.sum(AX[i2]*delta2) + 2*A[i2,i1]*np.sum(delta1*delta2)

        if diff > -tol * (maxstall * 1.0 / (iters + maxstall)):
            X[i1] = x2
            X[i2] = x1
            Ai = A[:, (i1, i2)]
            Deltas = np.vstack((delta2, delta1))
            AX -= np.dot(Ai, Deltas)

            cossim += diff / d
            updates += 1
            if updates > maxupdate:
                break

            stalls[istall] = stall
            stall = 0
            istall = (istall+1) % nstalls
            if stalls[:updates].mean() > maxstall:
                break

        # only report takes current time
        if iters % report == 0:
            t = time()
            if not silent: print "%d | %d | %.0f iters/min" % (iters, stalls[:updates].mean(), report*60.0/(t-tlast))
            tlast = t
            if t - t0 > maxtime:
                break

    if not silent: print "final similarity: ", cossim
    return X, cossim, iters, updates
















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












# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
from time import time


def elmvis(Xraw,
           A,
           slowdown=10,
           report=5,
           maxtime=24*60*60,
           tol=0,
           batch=None,
           maxiter=None,
           maxupdate=None,
           maxstall=None,
           cossim=None,
           silent=False):
    X = Xraw / np.linalg.norm(Xraw, axis=1)[:, None]  # unit-length version of X
    Xh = np.dot(A, X)  # X_hat, predicted value of X
    N, d = X.shape
    I = np.arange(N)  # index of samples

    # set default values
    if cossim is None: cossim = np.trace(X.T.dot(A).dot(X)) / N
    if batch is None:  # take 1% as a batch
        batch = int(N*0.01)
        batch = max(3, batch)
        batch = min(100, batch)
    if maxiter is None: maxiter = N*N*N
    if maxupdate is None: maxupdate = N*N
    if maxstall is None: maxstall = N*N

    if not silent:
        print "original similarity: ", cossim

    t0 = tlast = time()
    list1 = []
    list2 = []
    iters = 0
    stall = 0
    updates = 0
    updates_last = 0
    iters_last = 0
    ups_max = 0

    while (iters < maxiter) and (stall < maxstall):
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
        diff = A[i1,i1]*np.sum(delta1**2) + 2*np.sum(Xh[i1]*delta1) +\
               A[i2,i2]*np.sum(delta2**2) + 2*np.sum(Xh[i2]*delta2) +\
             2*A[i2,i1]*np.sum(delta1*delta2)

        if diff > tol:  # if found successful swap
            cossim += diff / N
            stall = 0

            if len(list1) >= batch or i1 in list1 or i1 in list2 or i2 in list1 or i2 in list2:  # apply batch
                x1 = X.take(list1, axis=0)
                x2 = X.take(list2, axis=0)
                X[list1] = x2
                X[list2] = x1

                tempI = I.take(list1)
                I[list1] = I[list2]
                I[list2] = tempI

                delta1 = x2 - x1
                delta2 = x1 - x2
                Ai = A.take(np.hstack((list1, list2)), axis=1)
                Xi = np.vstack((delta2, delta1))
                Xh -= np.dot(Ai, Xi)

                updates += len(list1)
                list1 = []
                list2 = []

                if updates >= maxupdate:
                    break

            list1.append(i1)
            list2.append(i2)

        t = time()
        if t - tlast > report:
            ups = (updates-updates_last)*1.0/(t-tlast)
            ips = (iters-iters_last)*1.0/(t-tlast)
            if not silent:
                print "%d iters | %d updates | %.0f iters/s | %.0f updates/s | cos similarity = %.4f" % (iters, updates, ips, ups, cossim)

            updates_last = updates
            iters_last = iters
            tlast = t
            ups_max = max(ups, ups_max)
            if ups < ups_max/slowdown:
                break

        if t - t0 > maxtime:
            break


    ips = iters*1.0/(time()-t0)

    # apply last batch update at exit
    if len(list1) > 0:
        x1 = X.take(list1, axis=0)
        x2 = X.take(list2, axis=0)
        X[list1] = x2
        X[list2] = x1

        tempI = I.take(list1)
        I[list1] = I[list2]
        I[list2] = tempI

        delta1 = x2 - x1
        delta2 = x1 - x2
        Ai = A.take(np.hstack((list1, list2)), axis=1)
        Xi = np.vstack((delta2, delta1))
        Xh -= np.dot(Ai, Xi)

        updates += len(list1)

    ups = updates*1.0/(time()-t0)
    Xraw[:] = Xraw[I]

    cossim = np.trace(X.T.dot(A).dot(X)) / N
    if not silent:
        print "final similarity: ", cossim

    info = {'cossim': cossim, 'iters': iters, 'updates': updates, 'ips': ips, 'ups': ups}
    return I, info

















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












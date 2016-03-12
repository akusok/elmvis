# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
cimport numpy as np
cimport cython
from time import time

cdef extern from "pdiff.h":
    long pdiff(double *A, double *X, double *AX, long *ari1, long *ari2, double tol, double *diff, long *iters, long d, long N, long rep)


@cython.boundscheck(False)
def elmvis(np.ndarray[np.float64_t, ndim=2] Xraw,
           np.ndarray[np.float64_t, ndim=2] A,
           double slowdown=10,
           long report=5,  # in seconds
           long maxtime=24*60*60,
           long searchbatch=200,
           double tol=0,
           long batch=0,
           long maxiter=0,
           long maxupdate=0,
           long maxstall=0,
           double cossim=0,
           int silent=0):
    """ELMVIS+ function in Cython.
    """
    cdef np.ndarray[np.float64_t, ndim=2] X = Xraw / np.linalg.norm(Xraw, axis=1)[:, None]
    cdef np.ndarray[np.long_t, ndim=1] I = np.arange(X.shape[0])
    cdef np.ndarray[np.long_t, ndim=1] tempI

    cdef long N = X.shape[0], d = X.shape[1]
    cdef long iters=0, updates=0, stall=0, iopt, updates_last=0, iters_last=0, searchiters, i1, i2
    cdef double t, t0, t1, tlast, diff, thr, ups, ips, ups_max=0, stop=0
    cdef np.ndarray[np.float64_t, ndim=2] Xh = np.dot(A, X)
    cdef np.ndarray[np.float64_t, ndim=2] Ai = np.empty((N, 2), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Xi = np.empty((2, d), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] x1
    cdef np.ndarray[np.float64_t, ndim=2] x2
    cdef np.ndarray[np.float64_t, ndim=2] delta1
    cdef np.ndarray[np.float64_t, ndim=2] delta2
    cdef np.ndarray[np.long_t, ndim=1] arri1 = np.ones(searchbatch, dtype=np.int_)
    cdef np.ndarray[np.long_t, ndim=1] arri2 = np.ones(searchbatch, dtype=np.int_)

    # set default values
    if cossim <= 0: cossim = np.trace(X.T.dot(A).dot(X)) / N
    if batch <= 0:
        batch = int(N*0.01)
        batch = max(3, batch)
        batch = min(100, batch)
    if maxiter <= 0: maxiter = N*N*N
    if maxupdate <= 0: maxupdate = N*N
    if maxstall <= 0: maxstall = N*N

    if not silent:
        print "original similarity: ", cossim

    t0 = tlast = time()
    list1 = []
    list2 = []

    while (iters < maxiter) and (stall < maxstall):
        arri1 = np.random.randint(0, N, size=searchbatch)
        arri2 = np.random.randint(0, N, size=searchbatch)

        iopt = pdiff(&A[0,0], &X[0,0], &Xh[0,0], <long *>arri1.data, <long *>arri2.data, tol, &diff, &searchiters, d, N, searchbatch)
        iters += searchiters
        stall += searchiters

        if iopt > -1:
            i1 = arri1[iopt]
            i2 = arri2[iopt]
            cossim += diff / N
            stall = 0

            if len(list1) >= batch or i1 in list1 or i1 in list2 or i2 in list1 or i2 in list2:
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























































# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
cimport numpy as np
cimport cython
from time import time

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cdef extern from "pdiff.h":
    long pdiff(double *A, double *X, double *AX, long *ari1, long *ari2, double tol, double *diff, long *iters, long d, long N, long rep)


'''
cdef long getdiff(double *A, double *X, double *AX, long *ari1, long *ari2, double tol, double *diff, long d, long N, long rep):
    """Does the same as the C function on top, but without parallelism.
    """
    cdef long r, j, i1, i2
    cdef double xi1, xi2, diff1
    for r in range(rep):
        i1 = ari1[r]
        i2 = ari2[r]
        diff1 = 0.0
        for j in range(d):
            xi1 = X[i1*d + j]
            xi2 = X[i2*d + j]
            diff1 += (A[i1*N + i1] * (xi2 - xi1) + 2*AX[i1*d + j]) * (xi2 - xi1)
            diff1 += (A[i2*N + i2] * (xi1 - xi2) + 2*(AX[i2*d + j] + A[i2*N + i1]*(xi2 - xi1))) * (xi1 - xi2)
        if diff1 > tol:
            diff[0] = diff1
            return r
    return -1
'''


@cython.boundscheck(False)
def elmvis(np.ndarray[np.float64_t, ndim=2] X,
           np.ndarray[np.float64_t, ndim=2] A,
           long batch = 500,
           double tol=1E-9,
           double cossim=0,
           long maxiter=1000000,
           long maxupdate=1000000,
           long maxtime=24*60*60,
           long report=1000,
           int silent=0):
    """ELMVIS+ function in Cython, double precision.
    """
    cdef long N = X.shape[0], d = X.shape[1]
    cdef long stall=0, iters=0, updates=0, nstalls=100, istall=0, i1, i2, iopt, lastreport, batchiters
    cdef double t, t0, t1, tlast, diff, thr, ipm=0
    cdef np.ndarray[np.float64_t, ndim=2] AX = np.dot(A, X)
    cdef np.ndarray[np.float64_t, ndim=2] Ai = np.empty((N, 2), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Xi = np.empty((2, d), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] x1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] x2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.long_t, ndim=1] arri1 = np.ones(batch, dtype=np.int_)
    cdef np.ndarray[np.long_t, ndim=1] arri2 = np.ones(batch, dtype=np.int_)
    cdef np.ndarray[np.float64_t, ndim=1] stalls = np.ones((nstalls,), dtype=np.float64) * batch

    assert batch > 1, "Batch size must be greater than one"

    if cossim == 0:
        cossim = np.trace(X.T.dot(A).dot(X)) / d
    if not silent: print "original similarity: ", cossim
    tol = tol*cossim

    t0 = tlast = time()
    lastreport = report
    while iters < maxiter:
        # get two different random numbers
        arri1 = np.random.randint(0, N, size=batch)
        arri2 = np.random.randint(0, N, size=batch)
        diff = 0
        batchiters = 0

        iopt = pdiff(&A[0,0], &X[0,0], &AX[0,0], <long *>arri1.data, <long *>arri2.data, tol, &diff, &batchiters, d, N, batch)
        if iopt == -1:
            iters = iters + batch
        else:
            iters = iters + batchiters
            i1 = arri1[iopt]
            i2 = arri2[iopt]
            x1 = X[i1].copy()
            x2 = X[i2].copy()
            X[i1] = x2
            X[i2] = x1
    
            delta1 = x2 - x1
            delta2 = x1 - x2          
            Ai = A.take((i1, i2), axis=1)
            Xi = np.vstack((delta2, delta1))
            AX -= np.dot(Ai, Xi)
 
            cossim += diff / d
            updates += 1
            if updates > maxupdate:
                break

            stalls[istall] = stall
            stall = 0
            istall = (istall+1) % nstalls

        # only report takes current time
        if iters > lastreport:
            while lastreport <= iters:
                lastreport += report
            t = time()
            ipm = report*60.0/(t-tlast)
            if not silent: print "%d | %d | %.0f iters/min" % (iters, stalls[:max(updates,1)].mean(), report*60.0/(t-tlast))
            tlast = t
            if t - t0 > maxtime:
                break

    if not silent: print "final similarity: ", cossim
    return X, cossim, iters, updates, ipm























###############################################################################
### same but 32-bit ###

cdef double getdiff32(float *A, float *X, float *AX, long d, long N, long i1, long i2):
    """Does the same as the C function on top, but without parallelism.
    """
    cdef long j, k
    cdef float xi1, xi2
    cdef double result=0
    for j in range(d):
        xi1 = X[i1*d + j]
        xi2 = X[i2*d + j]
        result += (A[i1*N + i1] * (xi2 - xi1) + 2*AX[i1*d + j]) * (xi2 - xi1)
        result += (A[i2*N + i2] * (xi1 - xi2) + 2*(AX[i2*d + j] + A[i2*N + i1]*(xi2 - xi1))) * (xi1 - xi2)
    return result


@cython.boundscheck(False)
def elmvis32(np.ndarray[np.float32_t, ndim=2] X,
           np.ndarray[np.float32_t, ndim=2] A,
           float tol=1E-9,
           float cossim=0,
           long maxiter=1000000,
           long maxstall=1000,
           long maxupdate=1000000,
           long maxtime=24*60*60,
           long report=1000,
           int silent=0):
    """ELMVIS+ function in Cython, single precision.
    """

    cdef long N = X.shape[0], d = X.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] AX = np.dot(A, X)
    cdef np.ndarray[np.float32_t, ndim=2] Ai = np.empty((N, 2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Xi = np.empty((2, d), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] x1 = np.empty(d, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] x2 = np.empty(d, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] delta1 = np.empty(d, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] delta2 = np.empty(d, dtype=np.float32)
    cdef long stall=0, iters=0, updates=0, nstalls=100, istall=0, i1, i2
    cdef double diff, t, t0, tlast
    cdef np.ndarray[np.float32_t, ndim=1] stalls = np.ones((nstalls,), dtype=np.float32)

    if cossim == 0:
        cossim = np.trace(X.T.dot(A).dot(X)) / d
    if not silent: print "original similarity: ", cossim
    tol = tol*cossim

    t0 = tlast = time()        
    while (iters < maxiter) and (stall < maxstall*10):
        iters = iters + 1
        stall = stall + 1

        # get two different random numbers
        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        diff = getdiff32(&A[0,0], &X[0,0], &AX[0,0], d, N, i1, i2)
        if diff > -tol * (maxstall * 1.0 / (iters + maxstall)):
            x1 = X[i1].copy()
            x2 = X[i2].copy()
            X[i1] = x2
            X[i2] = x1
    
            delta1 = x2 - x1
            delta2 = x1 - x2          
            Ai = A.take((i1, i2), axis=1)
            Xi = np.vstack((delta2, delta1))
            AX -= np.dot(Ai, Xi)
 
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
            if not silent: print "%d | %d | %.0f iters/min" % (iters, stalls[:max(updates,1)].mean(), report*60.0/(t-tlast))
            tlast = t
            if t - t0 > maxtime:
                break
    if not silent: print "final similarity: ", cossim
    return X, cossim, iters, updates


































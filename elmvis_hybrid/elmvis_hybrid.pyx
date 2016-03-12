# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:18:24 2015

@author: akusok
"""

import numpy as np
cimport numpy as np
cimport cython
from time import time

from pycuda import autoinit, gpuarray
from skcuda import linalg
from pycuda.compiler import SourceModule


cdef double getdiff(double *A, double *X, double *AX, long d, long N, long i1, long i2):
    """Does the same as the C function on top, but without parallelism.
    """
    cdef long j, k
    cdef double xi1, xi2, result=0
    for j in range(d):
        xi1 = X[i1*d + j]
        xi2 = X[i2*d + j]
        result += (A[i1*N + i1] * (xi2 - xi1) + 2*AX[i1*d + j]) * (xi2 - xi1)
        result += (A[i2*N + i2] * (xi1 - xi2) + 2*(AX[i2*d + j] + A[i2*N + i1]*(xi2 - xi1))) * (xi1 - xi2)
    return result


@cython.boundscheck(False)
def elmvis(np.ndarray[np.float64_t, ndim=2] X,
           np.ndarray[np.float64_t, ndim=2] A,
           double tol=1E-9,
           double cossim=0,
           long maxiter=1000000,
           long maxstall=1000,
           long maxupdate=1000000,
           long maxtime=24*60*60,
           long report=1000,
           int silent=0):
    """ELMVIS+ function in Cython with GPU update step, double precision.
    """

    cdef long N = X.shape[0], d = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] AX = np.dot(A, X)
    cdef np.ndarray[np.float64_t, ndim=2] Ai = np.empty((N, 2), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Xi = np.empty((2, d), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] x1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] x2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta2 = np.empty(d, dtype=np.float64)
    cdef long stall=0, iters=0, updates=0, nstalls=100, istall=0, i1, i2
    cdef double diff, t, t0, tlast
    cdef np.ndarray[np.float64_t, ndim=1] stalls = np.ones((nstalls,), dtype=np.float64)

    if cossim == 0:
        cossim = np.trace(X.T.dot(A).dot(X)) / d
    if not silent: print "original similarity: ", cossim
    tol = tol*cossim

    # init GPU
    dt = np.float64
    try:
        linalg.init()
    except ImportError as e:
        print e
    devA = gpuarray.to_gpu(A.astype(dt))
    devX = gpuarray.to_gpu(X.astype(dt))
    devAX = linalg.dot(devA, devX)
    devAi = gpuarray.empty((N, 2), dtype=dt)
    devDelta = gpuarray.empty((2, d), dtype=dt)
    
    t0 = tlast = time()        
    while (iters < maxiter) and (stall < maxstall*10):
        iters = iters + 1
        stall = stall + 1

        # get two different random numbers
        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        diff = getdiff(&A[0,0], &X[0,0], &AX[0,0], d, N, i1, i2)
        if diff > -tol * (maxstall * 1.0 / (iters + maxstall)):
            devAi[:, 0] = devA[:, i1]
            devAi[:, 1] = devA[:, i2]
            devDelta[0, :] = devX[i1, :] - devX[i2, :]
            devDelta[1, :] = devX[i2, :] - devX[i1, :]
            linalg.add_dot(devAi, devDelta, devAX, alpha=-1)
            devAX.get(ary=AX)

            x1 = X[i1].copy()
            x2 = X[i2].copy()
            X[i1] = x2
            X[i2] = x1
            devX[i1] = x2
            devX[i2] = x1
 
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
             double tol=1E-9,
             double cossim=0,
             long maxiter=1000000,
             long maxstall=1000,
             long maxupdate=1000000,
             long maxtime=24*60*60,
             long report=1000,
             int silent=0):
    """ELMVIS+ function in Cython with GPU update step, double precision.
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

    # init GPU
    dt = np.float32
    try:
        linalg.init()
    except ImportError as e:
        print e
    devA = gpuarray.to_gpu(A.astype(dt))
    devX = gpuarray.to_gpu(X.astype(dt))
    devAX = linalg.dot(devA, devX)
    devAi = gpuarray.empty((N, 2), dtype=dt)
    devDelta = gpuarray.empty((2, d), dtype=dt)
    
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
            devAi[:, 0] = devA[:, i1]
            devAi[:, 1] = devA[:, i2]
            devDelta[0, :] = devX[i1, :] - devX[i2, :]
            devDelta[1, :] = devX[i2, :] - devX[i1, :]
            linalg.add_dot(devAi, devDelta, devAX, alpha=-1)
            devAX.get(ary=AX)

            x1 = X[i1].copy()
            x2 = X[i2].copy()
            X[i1] = x2
            X[i2] = x1
            devX[i1] = x2
            devX[i2] = x1
 
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






































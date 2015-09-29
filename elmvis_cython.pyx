# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
cimport numpy as np
cimport cython
from multiprocessing import Queue
from cython.parallel import prange
from time import time
import os

# this was for parallel, but now GPU handles parallel
#cdef extern from "getdiff.h":
#    cdef double getdiff(double *A, double *Y, int d, int N, int i1, int i2, int nthr)


cdef double getdiff(double *A, double *Y, double *AY, int d, int N, int i1, int i2):
    """Does the same as the C function on top, but without parallelism.
    """
    cdef int j, k
    cdef double yi1, yi2, result=0
    for j in range(d):
        yi1 = Y[i1*d + j]
        yi2 = Y[i2*d + j]
        result += (A[i1*N + i1] * (yi2 - yi1) + 2*AY[i1*d + j]) * (yi2 - yi1)
        result += (A[i2*N + i2] * (yi1 - yi2) + 2*(AY[i2*d + j] + A[i2*N + i1]*(yi2 - yi1))) * (yi1 - yi2)
    return result


@cython.boundscheck(False)
def elmvis_cython(np.ndarray[np.float64_t, ndim=2] Yin,
                  np.ndarray[np.float64_t, ndim=2] A,
                  double tol = 1E-9,
                  int maxiter = 100000,
                  int maxstall = 10000,
                  int report = 1000):
    
    cdef int N = Yin.shape[0], d = Yin.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] Y = Yin.copy()
    cdef np.ndarray[np.float64_t, ndim=2] AY = np.dot(A, Yin)
    cdef np.ndarray[np.float64_t, ndim=2] Ai = np.empty((N, 2), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Yi = np.empty((2, d), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] y1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] y2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta2 = np.empty(d, dtype=np.float64)
    cdef int stall = 0, iters = 0, i1, i2
    print "original error:", np.diag(Y.T.dot(A).dot(Y)).sum() / d    
    
    t = time()
    while (iters < maxiter) and (stall < maxstall):
        iters = iters + 1
        stall = stall + 1

        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        if getdiff(&A[0,0], &Y[0,0], &AY[0,0], d, N, i1, i2) > -(tol * maxstall) / (iters + maxstall) :
            stall = 0
            y1 = Y[i1].copy()
            y2 = Y[i2].copy()
            Y[i1] = y2
            Y[i2] = y1
    
            delta1 = y2 - y1
            delta2 = y1 - y2          
            Ai = A.take((i1, i2), axis=1)
            Yi = np.vstack((delta2, delta1))
            AY -= np.dot(Ai, Yi)
 
        if iters % report == 0:
            print "%d | %d | %.0f iters/min" % (iters, stall, report*60.0/(time()-t))
            t = time()

    bests = np.diag(Y.T.dot(A).dot(Y)).sum() / d
    print "final error:", bests
    return Y, bests






































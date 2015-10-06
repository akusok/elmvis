# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:18:24 2015

@author: akusok
"""

import numpy as np
cimport numpy as np
cimport cython
from multiprocessing import Queue
from cython.parallel import prange
from time import time
import os

from pycuda import autoinit, gpuarray
from skcuda import linalg
from pycuda.compiler import SourceModule

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
def elmvis_hybrid(np.ndarray[np.float64_t, ndim=2] Yin,
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
    cdef double ips = 0, err
    err = np.diag(Y.T.dot(A).dot(Y)).sum() / d
    print "original error:", err
    tol *= err

    # init GPU
    try:
        linalg.init()
    except ImportError as e:
        print e
    devA = gpuarray.to_gpu(A)
    devY = gpuarray.to_gpu(Y)
    devAY = linalg.dot(devA, devY)
    devAi = gpuarray.empty((N, 2), dtype=np.float64)
    devDelta = gpuarray.empty((2, d), dtype=np.float64)
    
    t = time()
    while (iters < maxiter) and (stall < maxstall):
        iters = iters + 1
        stall = stall + 1

        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        if getdiff(&A[0,0], &Y[0,0], &AY[0,0], d, N, i1, i2) > -(tol * maxstall) / (iters + maxstall) :
            stall = 0

            devAi[:, 0] = devA[:, i1]
            devAi[:, 1] = devA[:, i2]
            devDelta[0, :] = devY[i1, :] - devY[i2, :]
            devDelta[1, :] = devY[i2, :] - devY[i1, :]
            linalg.add_dot(devAi, devDelta, devAY, alpha=-1)
            devAY.get(ary=AY)

            y1 = Y[i1].copy()
            y2 = Y[i2].copy()
            Y[i1] = y2
            Y[i2] = y1
            devY[i1] = y2
            devY[i2] = y1
 
        if iters % report == 0:
            ips = report*1.0/(time()-t)
            print "%d | %d | %.0f iters/min" % (iters, stall, report*60.0/(time()-t))
            t = time()

    bests = np.diag(Y.T.dot(A).dot(Y)).sum() / d
    print "final error:", bests
    return Y, bests, ips






































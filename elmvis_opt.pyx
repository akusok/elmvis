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

cdef extern from "getdiff.h":
    cdef double getdiff(double *A, double *Y, int d, int N, int i1, int i2, int nthr)


cdef double getdiff_cython(double *A, double *Y, int d, int N, int i1, int i2, int nthr):
    """Does the same as the C function on top, but without parallelism.
    """
    cdef int j, k
    cdef double t1, t2, y1, yi1, yi2, result=0
    for j in prange(d, nogil=True, num_threads=nthr):
        yi1 = Y[i1*d + j]
        yi2 = Y[i2*d + j]
        t1 = 0
        t2 = A[i2*N + i1] * (yi2 - yi1)
        for k in range(N):
            y1 = Y[k*d + j]
            t1 += A[i1*N + k] * y1
            t2 += A[i2*N + k] * y1
        result += (A[i1*N + i1] * (yi2 - yi1) + 2*t1) * (yi2 - yi1)
        result += (A[i2*N + i2] * (yi1 - yi2) + 2*t2) * (yi1 - yi2)
    return result


def bench(np.ndarray[np.float64_t, ndim=2] Yin,
        np.ndarray[np.float64_t, ndim=2] A,
        int maxiter,
        int nthreads = 1):
    cdef int N = Yin.shape[0], d = Yin.shape[1], i1 = 1, i2 = 2, j
    cdef double diff = 0
    
    t = time()
    for j in range(maxiter):
        diff += getdiff(&A[0,0], &Yin[0,0], d, N, i1, i2, nthreads)
    t = time()-t
    return t, diff


def opt(np.ndarray[np.float64_t, ndim=2] Yin,
        np.ndarray[np.float64_t, ndim=2] A,
        double tol = 1E-9,
        int maxiter = 100000,
        int maxstall = 10000,
        int report = 1000,
        int nthreads = 1):
    
    cdef int N = Yin.shape[0], d = Yin.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] Y = Yin.copy()
    cdef np.ndarray[np.float64_t, ndim=1] s = np.diag(Y.T.dot(A).dot(Y)).copy()
    cdef np.ndarray[np.float64_t, ndim=1] y1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] y2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] a1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] a2 = np.empty(d, dtype=np.float64)
 
    cdef double bests = s.sum(), diff, ips = 0
    cdef int stall = 0, iters = 0, i1, i2, j, k
    
    t = time()
    while (iters < maxiter) and (stall < maxstall):
        iters = iters + 1
        stall = stall + 1

        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        if getdiff(&A[0,0], &Y[0,0], d, N, i1, i2, nthreads) > -(tol * maxstall) / (iters + maxstall) :
            # full permute, initial algorithm of ELMVIS+
            # cannot merge steps because the method changes 'Y' on-the-fly
            for j in range(d):
                y1[j] = Y[i1, j]
                y2[j] = Y[i2, j]
                delta1[j] = y2[j] - y1[j]
                delta2[j] = y1[j] - y2[j]

            a1 = np.dot(A[i1, :], Y)
            for j in range(d):
                s[j] += (A[i1, i1]*delta1[j] + 2*a1[j])*delta1[j]
                Y[i1, j] = y2[j]
        
            a2 = np.dot(A[i2, :], Y)
            for j in range(d):
                s[j] += (A[i2, i2]*delta2[j] + 2*a2[j])*delta2[j]
                Y[i2, j] = y1[j]
    
            bests = s.sum()
            stall = 0
 
        if iters % report == 0:
            ips = report*1.0/(time()-t)
            print "%d | %d | %.0f iters/min" % (iters, stall, ips*60)
            t = time()

    return Y, bests, ips






cdef double getdiff2(double *A, double *Y, double *AY, int d, int N, int i1, int i2):
    """Does the same as the C function on top, but without parallelism.
    """
    cdef int j, k
    cdef double yi1, yi2, result=0
    for j in range(d):
        yi1 = Y[i1*d + j]
        yi2 = Y[i2*d + j]
        result += (A[i1*N + i1] * (yi2 - yi1) + 2*AY[i1*d + j]) * (yi2 - yi1)
        result += (A[i2*N + i2] * (yi1 - yi2) + 2*(AY[i2*d + j] + A[i2*N + i1])) * (yi1 - yi2)
    return result

@cython.boundscheck(False)
def op2(np.ndarray[np.float64_t, ndim=2] Yin,
        np.ndarray[np.float64_t, ndim=2] A,
        double tol = 1E-9,
        int maxiter = 100000,
        int maxstall = 10000,
        int report = 1000,
        int nthreads = 1):
    
    cdef int N = Yin.shape[0], d = Yin.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] Y = Yin.copy()
    cdef np.ndarray[np.float64_t, ndim=2] AY = np.dot(A, Yin)
    cdef np.ndarray[np.float64_t, ndim=2] Ai = np.empty((N, 2), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Yi = np.empty((2, d), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] s = np.diag(Y.T.dot(A).dot(Y)).copy()
    cdef np.ndarray[np.float64_t, ndim=1] y1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] y2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] a1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] a2 = np.empty(d, dtype=np.float64)
 
    cdef double bests = s.sum(), diff, ips = 0
    cdef int stall = 0, iters = 0, i1, i2, j, k
    
    t = time()
    while (iters < maxiter) and (stall < maxstall):
        iters = iters + 1
        stall = stall + 1

        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        if getdiff2(&A[0,0], &Y[0,0], &AY[0,0], d, N, i1, i2) > -(tol * maxstall) / (iters + maxstall) :
            # full permute, initial algorithm of ELMVIS+
            # cannot merge steps because the method changes 'Y' on-the-fly
            y1 = Y[i1].copy()
            y2 = Y[i2].copy()
            delta1 = y2 - y1
            delta2 = y1 - y2          

            for j in range(d):
                s[j] += (A[i1, i1]*delta1[j] + 2*AY[i1, j])*delta1[j]
                s[j] += (A[i2, i2]*delta2[j] + 2*(AY[i2, j] + A[i2, i1]*delta1[j]))*delta2[j]

            Y[i1] = y2
            Y[i2] = y1
    
            Ai = A.take((i1, i2), axis=1)
            Yi = np.vstack((delta2, delta1))
            AY -= np.dot(Ai, Yi)
            bests = s.sum()
            stall = 0
 
        if iters % report == 0:
            ips = report*1.0/(time()-t)
            print "%d | %d | %.0f iters/min" % (iters, stall, ips*60)
            t = time()

    return Y, bests, ips






































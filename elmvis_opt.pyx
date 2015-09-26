#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
cimport numpy as np
cimport cython
from time import time


#
#def permute(np.ndarray[np.float64_t, ndim=2] Y,
#            np.ndarray[np.float64_t, ndim=1] s,
#            np.ndarray[np.float64_t, ndim=2] A,
#            int i1, int i2):
#
#    cdef np.ndarray[np.float64_t, ndim=1] snew = s.copy()
#    cdef np.ndarray[np.float64_t, ndim=2] Ynew = Y.copy()
#    cdef np.ndarray[np.float64_t, ndim=1] y1 = Y[i1]
#    cdef np.ndarray[np.float64_t, ndim=1] y2 = Y[i2]
#    cdef np.ndarray[np.float64_t, ndim=1] delta1 = y2 - y1
#    cdef np.ndarray[np.float64_t, ndim=1] delta2 = y1 - y2    
#    cdef int d = Y.shape[1]   
#    cdef Py_ssize_t j
#
#    cdef np.ndarray[np.float64_t, ndim=1] a1 = np.dot(A[i1, :], Ynew)
#    for j in range(d):
#        snew[j] += (A[i1, i1]*delta1[j] + 2*a1[j])*delta1[j]
#        Ynew[i1, j] = y2[j]
#
#    cdef np.ndarray[np.float64_t, ndim=1] a2 = np.dot(A[i2, :], Ynew)
#    for j in range(d):
#        snew[j] += (A[i2, i2]*delta2[j] + 2*a2[j])*delta2[j]
#        Ynew[i2, j] = y1[j]
#
#    return snew, Ynew





def opt(np.ndarray[np.float64_t, ndim=2] Yin,
        np.ndarray[np.float64_t, ndim=2] A,
        int maxiter = 100000,
        int maxstall = 10000,
        int report = 1000):
    
    cdef int N = Yin.shape[0], d = Yin.shape[1], j    
    cdef np.ndarray[np.float64_t, ndim=2] Y = Yin.copy()
    cdef np.ndarray[np.float64_t, ndim=1] s = np.diag(Y.T.dot(A).dot(Y)).copy()
    cdef np.ndarray[np.float64_t, ndim=1] y1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] y2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta1 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] delta2 = np.empty(d, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] a1
    cdef np.ndarray[np.float64_t, ndim=1] a2
    cdef double swap, bests, sdiff, rootmstall = maxstall**0.5
 
    bests = s.sum()
    
    cdef int stall = 0, iters = 0, i1, i2
    t = time()
    while (iters < maxiter) and (stall < maxstall):
        stall = stall + 1
        iters = iters + 1

        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        # find effect of permutation
        sdiff = 0

        for j in range(d):
            y1[j] = Y[i1, j]
            y2[j] = Y[i2, j]
            delta1[j] = y2[j] - y1[j]
            delta2[j] = y1[j] - y2[j]

            
        a1 = np.dot(A[i1, :], Y)
        for j in range(d):
            sdiff += (A[i1, i1]*delta1[j] + 2*a1[j])*delta1[j]


        a2 = np.dot(A[i2, :], Y) + A[i2, i1]*delta1
        for j in range(d):
            sdiff += (A[i2, i2]*delta2[j] + 2*a2[j])*delta2[j]


        if sdiff > -rootmstall / (maxstall + iters):
            # full permute
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
            print "%d | %d | %.0f iters/min" % (iters, stall, report*60.0/(time()-t))
            t = time()

    return Y, bests





# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 02:27:40 2015

@author: akusok
"""


import numpy as np
from time import time

from pycuda import autoinit, gpuarray
from skcuda import linalg
from pycuda.compiler import SourceModule


def elmvis(X, A, tol=1E-9, cossim=None,
           maxiter=1000000,
           maxstall=1000,
           maxupdate=1000000,
           maxtime=24*60*60,
           report=1000,
           silent=False):
    """ELMVIS+ function running in GPU memory.
    """

    N, d = X.shape
    if cossim is None:
        cossim = np.trace(X.T.dot(A).dot(X)) / d
    if not silent: print "original similarity: ", cossim
    tol = tol*cossim

    # init GPU
    dt = X.dtype.type
    try:
        linalg.init()
    except ImportError as e:
        print e
    devA = gpuarray.to_gpu(A.astype(dt))
    devX = gpuarray.to_gpu(X.astype(dt))
    devXi1 = gpuarray.empty((d,), dtype=dt)
    devAX = linalg.dot(devA, devX)
    devAi = gpuarray.empty((N, 2), dtype=dt)
    devDelta = gpuarray.empty((2, d), dtype=dt)
    result = gpuarray.empty((d,), dtype=dt)

    # fast kernel for a better solution search
    kernel = """
        __global__ void diff(%s *A, %s *Y, %s *AY, %s *result, long d, long N, long i1, long i2) {
            long j = blockDim.x * blockIdx.x + threadIdx.x;
            %s yi1 = Y[i1*d + j];
            %s yi2 = Y[i2*d + j];
            result[j] = (A[i1*N + i1] * (yi2 - yi1) + 2*AY[i1*d + j]) * (yi2 - yi1) +
                        (A[i2*N + i2] * (yi1 - yi2) + 2*(AY[i2*d + j] + A[i2*N + i1]*(yi2 - yi1))) * (yi1 - yi2);
        }
        """
    if dt is np.float64:
        kernel = kernel % ("double", "double", "double", "double", "double", "double")
    else:
        kernel = kernel % ("float", "float", "float", "float", "float", "float")
    mod_diff = SourceModule(kernel)
    dev_diff = mod_diff.get_function("diff")
    dev_diff.prepare("PPPPllll")
    block = result._block
    grid = (int(np.ceil(1.0 * result.shape[0] / block[0])), 1)

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

        dev_diff.prepared_call(grid, block, devA.gpudata, devX.gpudata, devAX.gpudata, result.gpudata, d, N, i1, i2)
        diff = np.sum(result.get())

        if diff > -tol * (maxstall * 1.0 / (iters + maxstall)):
            devAi[:, 0] = devA[:, i1]
            devAi[:, 1] = devA[:, i2]
            devDelta[0, :] = devX[i1, :] - devX[i2, :]
            devDelta[1, :] = devX[i2, :] - devX[i1, :]
            linalg.add_dot(devAi, devDelta, devAX, alpha=-1)

            devXi1[:] = devX[i1, :]
            devX[i1] = devX[i2]
            devX[i2] = devXi1

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

    X = devX.get()
    if not silent: print "final similarity: ", cossim
    return X, cossim, iters, updates







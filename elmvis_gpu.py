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



def elmvis_gpu(Y, A, tol=1E-9, maxiter=100000, maxstall=10000, report=1000, dt=np.float64):
    N, d = Y.shape
    print "original error:", np.diag(Y.T.dot(A).dot(Y)).sum() / d

    # init GPU
    linalg.init()
    devA = gpuarray.to_gpu(A.astype(dt))
    devY = gpuarray.to_gpu(Y.astype(dt))
    devYi1 = gpuarray.empty((d,), dtype=dt)
    devAY = linalg.dot(devA, devY)
    devAi = gpuarray.empty((N, 2), dtype=dt)
    devDelta = gpuarray.empty((2, d), dtype=dt)
    result = gpuarray.empty((d,), dtype=dt)

    # fast search kernel
    kernel = """
        __global__ void diff(%s *A, %s *Y, %s *AY, %s *result, int d, int N, int i1, int i2) {
            unsigned j = blockDim.x * blockIdx.x + threadIdx.x;
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
    dev_diff.prepare("PPPPiiii")
    block = result._block
    grid = (int(np.ceil(1.0 * result.shape[0] / block[0])), 1)

    stall = 0
    iters = 0
    t = time()
    while (stall < maxstall) and (iters < maxiter):
        iters += 1
        stall += 1

        i1, i2 = np.random.randint(0, N, size=2)
        while i1 == i2:
            i1, i2 = np.random.randint(0, N, size=2)

        dev_diff.prepared_call(grid, block, devA.gpudata, devY.gpudata, devAY.gpudata, result.gpudata, d, N, i1, i2)

        if np.sum(result.get()) > -(tol * maxstall) / (iters + maxstall):
            stall = 0

            devAi[:, 0] = devA[:, i1]
            devAi[:, 1] = devA[:, i2]
            devDelta[0, :] = devY[i1, :] - devY[i2, :]
            devDelta[1, :] = devY[i2, :] - devY[i1, :]
            linalg.add_dot(devAi, devDelta, devAY, alpha=-1)

            devYi1[:] = devY[i1, :]
            devY[i1] = devY[i2]
            devY[i2] = devYi1

        if iters % report == 0:
            print "%d | %d | %.0f iters/min" % (iters, stall, report*60.0/(time()-t))
            t = time()

    Y = devY.get()
    bests = np.diag(Y.T.dot(A).dot(Y)).sum() / d
    return Y, bests











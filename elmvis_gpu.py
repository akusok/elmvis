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
    """ELMVIS+ function running in GPU memory.
    """
    X = Xraw / np.linalg.norm(Xraw, axis=1)[:, None]  # unit-length version of X
    Xh = np.dot(A, X)  # X_hat, predicted value of X
    N, d = X.shape
    I = np.arange(N)  # index of samples

    # set default values
    if cossim is None: cossim = np.trace(X.T.dot(A).dot(X)) / N
    if maxiter is None: maxiter = N*N*N
    if maxupdate is None: maxupdate = N*N
    if maxstall is None: maxstall = N*N

    if not silent:
        print "original similarity: ", cossim

    # init GPU
    dt = X.dtype.type
    try:
        linalg.init()
    except ImportError as e:
        print e
    devA = gpuarray.to_gpu(A.astype(dt))
    devX = gpuarray.to_gpu(X.astype(dt))
    devXi1 = gpuarray.empty((d,), dtype=dt)
    devXh = linalg.dot(devA, devX)
    devAi = gpuarray.empty((N, 2), dtype=dt)
    devDelta = gpuarray.empty((2, d), dtype=dt)
    result = gpuarray.empty((d,), dtype=dt)

    # swap kernel
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

    t0 = tlast = time()
    stall = 0
    iters = 0
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

        dev_diff.prepared_call(grid, block, devA.gpudata, devX.gpudata, devXh.gpudata, result.gpudata, d, N, i1, i2)
        diff = np.sum(result.get())

        if diff > tol:
            stall = 0
            devAi[:, 0] = devA[:, i1]
            devAi[:, 1] = devA[:, i2]
            devDelta[0, :] = devX[i1, :] - devX[i2, :]
            devDelta[1, :] = devX[i2, :] - devX[i1, :]
            linalg.add_dot(devAi, devDelta, devXh, alpha=-1)

            tI = I[i1]
            I[i1] = I[i2]
            I[i2] = tI

            devXi1[:] = devX[i1, :]
            devX[i1] = devX[i2]
            devX[i2] = devXi1

            cossim += diff / N
            updates += 1
            if updates > maxupdate:
                break

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
    ups = updates*1.0/(time()-t0)
    Xraw[:] = Xraw[I]

    cossim = np.trace(X.T.dot(A).dot(X)) / N
    if not silent:
        print "final similarity: ", cossim

    info = {'cossim': cossim, 'iters': iters, 'updates': updates, 'ips': ips, 'ups': ups}
    return I, info







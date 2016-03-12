# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:20:36 2015

@author: akusok
"""

import numpy as np
import hpelm
from matplotlib import pyplot as plt, cm, colors

from elmvis_cython import elmvis
#from elmvis_python import elmvis
#from elmvis_gpu import elmvis

def run():
    X = np.loadtxt("mnist_Xts.txt")
    Y = np.loadtxt("mnist_Yts.txt")
    X = X[:, (X > 0).sum(axis=0) >= 5].copy()  # choose features with at least 5 non-zero entrances
    N, d = X.shape

    V = np.random.rand(N, 2)*2 - 1
    V = (V - V.mean(0)) / V.std(0)

    # build initial ELM
    L = 18
    print "L", L
    elm = hpelm.ELM(2, d)
    elm.add_neurons(2, 'lin')
    elm.add_neurons(L, 'sigm')
    H = elm.project(V)
    A = H.dot(np.linalg.pinv(np.dot(H.T, H))).dot(H.T)

    I, info = elmvis(X, A, slowdown=20)
    print info

    scalarMap = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=10), cmap=plt.get_cmap('gist_rainbow'))
    clr = lambda c: scalarMap.to_rgba(c)

    for i in range(X.shape[0]):
        plt.text(V[i,0], V[i,1], "%d"%Y[I[i]], size=8, color=clr(Y[I[i]]))

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig("plot_mnist.pdf", bbox_inches="tight")

    plt.show()



if __name__ == "__main__":
    run()
    print "Done"
















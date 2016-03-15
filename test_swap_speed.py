import numpy as np
from matplotlib import pyplot as plt

import hpelm
from elmvis_cython import elmvis as elmvis_c
from elmvis_python import elmvis as elmvis_p
from elmvis_gpu import elmvis as elmvis_g


def runswaps():
    X = np.random.rand(1000, 10000)
    N, d = X.shape

    V = np.random.rand(N, 2)*2 - 1
    V = (V - V.mean(0)) / V.std(0)

    swaps = []
    i = 0
    for d in range(20, 1001, 20):
        elm = hpelm.ELM(2, d)
        elm.add_neurons(2, 'lin')
        elm.add_neurons(20, 'sigm')
        H = elm.project(V)

        A = H.dot(np.linalg.pinv(np.dot(H.T, H))).dot(H.T)

        X1 = X[:, :d].copy()
        _, info = elmvis_c(X1, A, tol=100, maxiter=10000, maxstall=10000, silent=1)
        ips_c = info['ips']

        X1 = X[:, :d].copy()
        _, info = elmvis_p(X1, A, tol=100, maxiter=10000, maxstall=10000, silent=1)
        ips_p = info['ips']

        X1 = X[:, :d].copy()
        _, info = elmvis_g(X1, A, tol=100, maxiter=10000, maxstall=10000, silent=1)
        ips_g = info['ips']

        swaps.append((ips_c, ips_p, ips_g))
        print d

    swaps = np.array(swaps)
    d = range(20, 1001, 20)
    plt.plot(d, swaps[:,0], '--r', linewidth=3, alpha=0.7)
    plt.plot(d, swaps[:,1], '--k')
    plt.plot(d, swaps[:,2], '--g')
    plt.ylabel("swaps per second")

    plt.legend(("C optimized", "Python", "GPU"))
    plt.title("1000 data points with $d$ features")
    plt.xlabel("$d$")
    plt.xlim([0, 1000])
    #plt.ylim([0, 23000000])
    #plt.yticks([0, 1000000, 5000000, 10000000, 15000000, 20000000, 22000000], ["0", "1 mln.", "5 mln.", "10 mln.", "15 mln.", "20 mln.", "22 mln."])
    plt.savefig("plot_ips.pdf", bbox_inches="tight")
    print "done"
    plt.show()


runswaps()
print "done"

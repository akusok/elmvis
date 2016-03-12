import numpy as np
from matplotlib import pyplot as plt

import hpelm
from elmvis_cython import elmvis as elmvis_c
from elmvis_python import elmvis as elmvis_p
from elmvis_gpu import elmvis as elmvis_g


def runupdates():
    X = np.random.rand(5000, 1000)
    N, d = X.shape

    V = np.random.rand(N, 2)*2 - 1
    V = (V - V.mean(0)) / V.std(0)

    updates = []
    i = 0
    for N in range(100, 5001, 100):
        elm = hpelm.ELM(2, d)
        elm.add_neurons(2, 'lin')
        elm.add_neurons(50, 'sigm')
        H = elm.project(V[:N])

        A = H.dot(np.linalg.pinv(np.dot(H.T, H))).dot(H.T)

        X1 = X[:N].copy()
        _, info = elmvis_c(X1, A, tol=-100, maxupdate=1000, cossim=1, silent=1)
        ups_c = info['ups']

        X1 = X[:N].copy()
        _, info = elmvis_p(X1, A, tol=-100, maxupdate=1000, cossim=1, silent=1)
        ups_p = info['ups']

        X1 = X[:N].copy()
        _, info = elmvis_p(X1, A, tol=-100, maxupdate=1000, batch=1, cossim=1, silent=1)
        ups_p1 = info['ups']

        X1 = X[:N].copy()
        _, info = elmvis_g(X1, A, tol=-100, maxupdate=1000, cossim=1, silent=1)
        ups_g = info['ups']

        updates.append((ups_c, ups_p, ups_p1, ups_g))
        print "%d/5000" % N

    updates = np.array(updates)
    N = range(100, 5001, 100)
    plt.plot(N, updates[:,0], '--r', linewidth=3, alpha=0.7)
    plt.plot(N, updates[:,1], '--k')
    plt.plot(N, updates[:,2], '-k')
    plt.plot(N, updates[:,3], '-g')
    plt.ylabel("updates per second")

    plt.legend(("C opt, 1% batch", "Python, 1% batch", "Python", "GPU"))
    plt.title("$N$ data points with 1000 features")
    plt.xlabel("$N$")
    plt.xlim([0, 5000])
    #plt.ylim([0, 23000000])
    #plt.yticks([0, 1000000, 5000000, 10000000, 15000000, 20000000, 22000000], ["0", "1 mln.", "5 mln.", "10 mln.", "15 mln.", "20 mln.", "22 mln."])
    plt.savefig("plot_ups.pdf", bbox_inches="tight")
    print "done"
    plt.show()


runupdates()
print "done"
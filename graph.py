# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:16:03 2015

@author: akusok
"""

import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("hybriddata_d3.txt")
d = data[:, 0]
p1 = data[:, 1]
p2 = data[:, 2]
c1 = data[:, 3]
c2 = data[:, 4]
g1 = data[:, 5]
g2 = data[:, 6]
h1 = data[:, 7]
h2 = data[:, 8]

kind = 2

if kind == 1:
    plt.plot(d, h1, '-r',  linewidth=3, alpha=0.7)
    plt.plot(d, p1, '-k')
    plt.plot(d, c1, '-b')
    plt.plot(d, g1, '-g')
    plt.ylabel("updates per second")
else:
    plt.plot(d, h2, '--r', linewidth=3, alpha=0.7)
    plt.plot(d, p2, '--k')
    plt.plot(d, c2, '--b')
    plt.plot(d, g2, '--g')
    plt.ylabel("swaps per second")


plt.legend(("gpu+c", "python", "c", "gpu"))
plt.title("$N$ data points with 3 features")
#plt.yscale('log')
plt.xlabel("$N$")
plt.xlim([0, 5000])
plt.ylim([0, 1690000])
plt.savefig("varN-d3_swaps_a.pdf", bbox_inches="tight")
print "done"
plt.show()
    

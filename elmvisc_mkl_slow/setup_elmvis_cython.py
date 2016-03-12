# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:21:14 2015

@author: akusok
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("elmvis_mkl",
                         ["elmvis_cython.pyx", "pdiff.c"],
                         extra_compile_args=['-DMKL_ILP64 -m64 -fopenmp'],
                         extra_link_args=[],
                         include_dirs=[numpy.get_include(), '/opt/intel/mkl/include'],
                         library_dirs=['/opt/intel/mkl/lib/intel64'],
                         libraries=['mkl_intel_ilp64', 'mkl_core', 'mkl_sequential', 'pthread', 'm', 'dl']
                         )]

setup(
  name = 'ELMVIS+ cython app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

print
print "Done!"
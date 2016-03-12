# ELMVIS+

These are 3 versions of ELMVIS+ code in Python, all impementing the same `elmvis` function. For details, check my updated paper "ELMVIS+: Fast Nonlinear Visualization Technique based on Cosine Distance and Extreme Learning Machines"


## Installation

Python version needs Numpy and works straight away.

Cython version (RECOMMENDED) needs Numpy, Cython and some OpenMP installed. Compile it with `python setup_elmvis_cython.py build_ext --inplace` to get `elmvis_cython.so` library. You can later import function from this library as `from elmvis_cython import elmvis`.

GPU version requires Numpy, Pycuda and Scikit-CUDA. It works in single and double precision, taking it from matrix X.


## Basic usage:
`elmvis(X, A, slowdown)`

Inputs:
* X: original data (like MNIST digits), an N-times-d matrix where 'N' is number of data samples and 'd' is data dimensionality.
* A: parameter matrix from ELM model (check paper), is (N*N) matrix.
* slowdown: stops the method when a number of successful updates per second decreases by this factor. Default is 10, deep optimization will be 100.

Returns:
* the method directly chages rows in X matrix in an optimal way. It can be used several times - improving the results each time.


## Advanced usage:
`I, info = elmvis(X, A, tol, slowdown, report, maxtime, batch, searchbatch, maxiter, maxupdate, maxstall, cossim, silent)`

Inputs:
* X: original data (like MNIST digits), an N-times-d matrix where 'N' is number of data samples and 'd' is data dimensionality.
* A: parameter matrix from ELM model (check paper), is (N*N) matrix.
* tol: tolerance of optimization method (like simulated annealing), default is zero.
* slowdown: stops the method when a number of successful updates per second decreases by this factor.
* report (seconds): number of seconds between reports of method performance; also `slowdown` is checked only during report.
* maxtime (seconds): maximum runtime after that the method stops, default is 1 day.
* batch: maximum size of an update batch, default is 0.01*N. Batch update is done immediately if the method sees repeting indexes, so it should work with large maximum batch sizes as well. In GPU implementation the batch is always one.
* searchbatch: batch size for parallel swap function to reduce function call overhead (see `pdiff.c`), the function checks this many random swaps and exit as soon as it finds a successful one. Default is 200, probably don't need to change that.
* maxiter: maximum number of swap steps the method will run, default is N^3.
* maxupdate: maximum number of update steps the method will do, actual updates can be a bit more due to batch system, default is N^2.
* maxstall: maximum number of swap steps without a single good update found, default is N^2.
* cossim: initial cosine similarity, default is to compute the cosine similarity from the data.
* silent: whether to report anything, default is True.

Returns:
* I: index of rows in the original X. The same index is already applied to rows of the input matrix X in-place.
* info: a dictionary with method statistics, like a total number of swaps/updates, and average swap/update per second.
* the method chages rows in X matrix in an optimal way.


## Testing ELMVIS+

There are three scripts to test ELMVIS+. They require HPELM toolbox for ELM, install it as `pip install hpelm`; and the Matplotlib package for plotting.

`test_MNIST.py` performs visualization of MNIST digits (from the included text files) on uniformly selected random points in two dimentions, and plots the corresponding numbers in different color. You can choose between different ELMVIS+ implementations by un-commenting import lines at the beginning of the file.

`test_update_speed.py` and `test_swap_speed.py` scripts test all three methods in update and swap speed, showing plots similar to the ones in the paper.

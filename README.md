## elmvis

These are 3 versions of ELMVIS+ code in Python, all impementing the same `elmvis` function. For details, check my updated paper "ELMVIS+: Fast Nonlinear Visualization Technique based on Cosine Distance and Extreme Learning Machines"

# Basic usage:
`elmvis(X, A, tol, slowdown)`

Inputs:
* X: original data (like MNIST digits), an N-times-d matrix where 'N' is number of data samples and 'd' is data dimensionality
* A: parameter matrix from ELM model (check paper), is (N*N) matrix
* tol: tolerance of optimization method (like simulated annealing), default is zero
* slowdown: stops the method when a number of successful updates per second decreases this much

Returns:
* the method chages rows in X matrix in an optimal way


# Advanced usage:
`I, info = elmvis(X, A, tol, slowdown, report, maxtime, batch, searchbatch, maxiter, maxupdate, maxstall, cossim, silent)`

Inputs:
* X: original data (like MNIST digits), an N-times-d matrix where 'N' is number of data samples and 'd' is data dimensionality
* A: parameter matrix from ELM model (check paper), is (N*N) matrix
* tol: tolerance of optimization method (like simulated annealing), default is zero
* slowdown: stops the method when a number of successful updates per second decreases this much
* report (in seconds): number of seconds between reports of method performance; also `slowdown` is checked only at report


Returns:
* the method chages rows in X matrix in an optimal way




* cossim: initial cosine similarity if available; the function will compute it if not provided. Provide anything non-zero to avoid that computation step
* maxiter: maximum number of swaps
* maxstall: maximum average number of swaps between updates, computed on the last 100 updates. Method finishes if no improvement is found within 10*maxstall swaps
* maxupdate: maximum number of updates
* maxtime: maximum runtime; runtime is checked only on updates
* report: report current progress every "report" number of swaps
* silent: suppress output

Returns:
* Xnew: data matrix with re-arranged rows, better suited for the visualization used
* cossimnew: new value of cosine similarity
* iters: how many iterations the method run
* updates: how many updates were performed

Hybrid and Cython implementations need to be compiled with Cython: use `python setup_elmvis_<hybrid or cython>.py build_ext --inplace`.
GPU code requires Pycuda and Skcuda, supports NVidia cards (tested on GTX Titan Black, speedup up to 30x with high-dimensional data).

Some methods have "emlvis32" function with the same syntax, it helps saving memory for large N (and large A, for example if N=60000, even 32-bit float matrix A takes 13.5GB RAM). For 'elmvis_gpu', the choise between single- and double-precision is done according to data type of matrix X (whether numpy.float32 or numpy.float64). Also, use 32-bit for GPUs with slow double precision support, like all Nvidia Maxwell series.

"""
It generates n random samples from the array a. If a is an integer, it generates n random samples from the range [1:a].
"""


import numpy as np

def randomsample(a, n):
    # If a is a scalar, treat it as the length of the array
    if np.isscalar(a):
        a = np.arange(1, a + 1)

    npts = len(a)

    if npts < n:
        raise ValueError(f'Trying to select {n} items from a list of length {npts}')

    item = np.zeros(n, dtype=int)

    for i in range(n):
        # Generate random value in the appropriate range
        r = np.random.randint(npts - i)
        item[i] = a[r]  # Select the rth element from the list
        a[r] = a[npts - i - 1]  # Overwrite selected element

    return item

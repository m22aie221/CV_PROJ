"""
The provided lscal function computes the color correction matrix M using the linear least squares method. Here's the equivalent Python implementation:
This Python implementation uses NumPy's linalg.lstsq function to solve the linear least squares problem. It takes the transpose of the input matrices rgb and xyz to match the dimensions expected by the function. Then, it computes the color correction matrix M using the least squares method and returns it.
"""

import numpy as np


def lscal(rgb, xyz):
    # Compute the color correction matrix using linear least squares method
    M = np.linalg.lstsq(rgb.T, xyz.T, rcond=None)[0].T
    return M

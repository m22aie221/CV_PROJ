"""
This Python function alshomocal performs the same functionality as its MATLAB counterpart. It estimates the color homography matrix for shading-independent color correction using the input RGB and XYZ data. The function uses the least squares method to find the transformation matrix.
"""

import numpy as np
from scipy.linalg import lstsq

def alshomocal(rgb, xyz):
    """
    ALSHOMOCAL estimates a color homography matrix for shading-independent
    color correction. This version is without outlier detection.

    Parameters:
    RGB: Input RGB with shadings
    XYZ: Input XYZ ground truth

    Returns:
    M: Color correction matrix
    """

    # Transpose RGB and XYZ matrices
    rgb_t = rgb.T
    xyz_t = xyz.T

    # Compute the least squares solution
    M, _, _, _ = lstsq(rgb_t, xyz_t)

    return M

# Example usage:
# M = alshomocal(rgb_data, xyz_data)

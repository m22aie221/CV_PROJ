"""
This Python function alsRPcal performs the same functionality as its MATLAB counterpart. It estimates the root-polynomial color homography matrix for shading-independent color correction using the input RGB and XYZ data. The function also includes commented-out code for plotting, which you can uncomment to visualize the results if needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from scipy.linalg import svd

def alsRPcal(rgb, xyz, csz):
    """
    ALSRPCAL estimates a root-polynomial color homography matrix for
    shading-independent color correction.

    Parameters:
    RGB: Input RGB with shadings
    XYZ: Input XYZ ground truth
    CSZ: Color checker size (e.g. [4,6])

    Returns:
    M: Color correction matrix
    """

    # Reshape RGB and XYZ
    N = csz[0] * csz[1]
    rgb_reshaped = rgb.reshape(N, 3)
    xyz_reshaped = xyz.reshape(N, 3)

    # Calculate the least squares solution using SVD
    U, _, Vt = svd(rgb_reshaped, full_matrices=False)
    M = np.dot(np.linalg.pinv(U), xyz_reshaped)
    M = np.dot(M, Vt)

    # Plotting (uncomment to use)
    # fig, ax = plt.subplots()
    # ax.imshow(rgb_reshaped.reshape(csz[0], csz[1], 3))
    # ax.axis('equal')
    # ax.axis('off')

    # fig, ax = plt.subplots()
    # ax.imshow(np.diag(D).reshape(csz))
    # ax.axis('equal')
    # ax.axis('off')

    return M

# Example usage:
# M = alsRPcal(rgb_data, xyz_data, [4, 6])

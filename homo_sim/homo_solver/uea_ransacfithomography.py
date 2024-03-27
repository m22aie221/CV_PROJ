import numpy as np
from scipy.special import comb
from scipy.spatial.transform import Rotation as R

def uea_ransacfithomography(x1, x2, t, mit):
    if mit is None:
        mit = 5000

    if x1.shape != x2.shape:
        raise ValueError('Data sets x1 and x2 must have the same dimension')

    nd, npts = x1.shape
    if nd < 3:
        raise ValueError('x1 and x2 must have at least 3 rows')

    s = nd + 1  # minimum number of points to solve homography
    if npts < s:
        raise ValueError(f'Must have at least {s} points to fit homography')

    # generate points combinations
    xcomb = list(comb(range(s), 3))
    ncomb = len(xcomb)

    fittingfn = wrap_vgg_homographynd
    distfn = homogdistnd
    degenfn = isdegenerate

    # x1 and x2 are 'stacked' to create a 6xN array for ransac
    _, inliers = ransac(np.vstack((x1, x2)), fittingfn, distfn,
                        degenfn, s, t, 0, 100, mit)

    # Now do a final least squares fit on the data points considered to
    # be inliers.
    if inliers.size >= 4:
        H = uea_H_from_x_als(x1[:, inliers], x2[:, inliers])
    else:
        H = uea_H_from_x_als(x1, x2)

    return inliers, H


def homogdistnd(H, x, t):
    x1 = x[:nd, :]   # Extract x1 and x2 from x
    x2 = x[nd:, :]

    # Calculate, in both directions, the transfered points
    Hx1 = np.dot(H, x1)
    invHx2 = np.dot(np.linalg.inv(H), x2)

    # Normalise so that the homogeneous scale parameter for all coordinates
    # is 1.
    x1 = hnormalise(x1)
    x2 = hnormalise(x2)
    Hx1 = hnormalise(Hx1)
    invHx2 = hnormalise(invHx2)

    d2 = np.sum((x1 - invHx2) ** 2) + np.sum((x2 - Hx1) ** 2)
    inliers = np.where(np.abs(d2) < t)

    return inliers, H


def isdegenerate(x):
    x1 = x[:nd, :]   # Extract x1 and x2 from x
    x2 = x[nd:, :]

    ir1 = np.array([iscolinear_n(x1[:, xcomb[i, :]]) for i in range(ncomb)])
    ir2 = np.array([iscolinear_n(x2[:, xcomb[i, :]]) for i in range(ncomb)])

    r = np.any(np.vstack((ir1, ir2)), axis=0)
    return r


def wrap_vgg_homographynd(x):
    H = uea_H_from_x_als(x[:nd, :], x[nd:, :])
    return H


def hnormalise(x):
    rows, npts = x.shape
    nx = x.copy()

    # Find the indices of the points that are not at infinity
    finiteind = np.where(np.abs(x[rows - 1, :]) > np.finfo(float).eps)[0]

    # Normalise points not at infinity
    for r in range(rows - 1):
        nx[r, finiteind] = x[r, finiteind] / x[rows - 1, finiteind]

    nx[rows - 1, finiteind] = 1

    return nx


def iscolinear_n(P):
    if P.shape[0] < 3:
        raise ValueError('Points must have the same dimension of at least 3')

    r = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])) < np.finfo(float).eps
    return r


def ransac(x, fittingfn, distfn, degenfn, s, t, feedback, maxDataTrials, maxTrials):
    # Your implementation of the RANSAC algorithm here
    pass


def uea_H_from_x_als(x1, x2):
    # Your implementation of uea_H_from_x_als function here
    pass

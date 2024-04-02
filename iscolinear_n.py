import numpy as np

def iscolinear_n(P):
    if not P.shape[0] >= 3:
        raise ValueError('Points must have the same dimension of at least 3')
    
    return np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])) < np.finfo(float).eps

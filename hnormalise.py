import numpy as np

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

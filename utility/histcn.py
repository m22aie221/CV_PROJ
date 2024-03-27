import numpy as np

def histcn(X, *edges, AccumData=None, Fun=None):
    """
    Compute n-dimensional histogram.

    Parameters:
        X: numpy array
            An (M x N) array representing M data points in R^N.
        edges: array-like
            Bin vectors on dimensions k, k=1...N. If it is a scalar (Nk), the bins
            will be the linear subdivision of the data on the range [min(X[:,k]), max(X[:,k])]
            into Nk sub-intervals. If it's empty, a default of 32 subdivisions will be used.
        AccumData: numpy array, optional
            An M x 1 array. Each value AccumData[k] corresponds to position X[k,:]
            and will be accumulated in the cell containing X. Default is None.
        Fun: function, optional
            A function that accepts a column vector and returns a numeric, logical,
            or char scalar, or a scalar cell. Default is None.

    Returns:
        count: numpy array
            n-dimensional array count of X on the bins.

        edges: list
            A list of length N, each providing the effective edges used in the respective dimension.

        mid: list
            A list of length N, each providing the mid points of the cellpatch used in the respective dimension.

        loc: numpy array
            Index location of X in the bins. Points have out of range coordinates will have zero at the corresponding dimension.
    """
    if X.ndim > 2:
        raise ValueError('X requires to be an (M x N) array of M points in R^N')
    
    DEFAULT_NBINS = 32

    # If edges are not provided, set them to DEFAULT_NBINS
    if len(edges) < X.shape[1]:
        edges = list(edges) + [DEFAULT_NBINS] * (X.shape[1] - len(edges))
    
    # Allocation of array loc: index location of X in the bins
    loc = np.zeros(X.shape)
    sz = np.zeros(X.ndim)

    # Loop over dimensions
    for d in range(X.shape[1]):
        ed = edges[d]
        Xd = X[:, d]

        if isinstance(ed, (int, float)):  # automatic linear subdivision
            ed = np.linspace(np.min(Xd), np.max(Xd), int(ed) + 1)
        edges[d] = ed

        # Call numpy's digitize on this dimension
        loc[:, d] = np.digitize(Xd, ed, right=False)
        sz[d] = len(ed) - 1
    
    # Compute the mid points
    mid = [0.5 * (e[:-1] + e[1:]) for e in edges]

    # This is needed for points that hit the right border
    mloc = np.max(loc, axis=0)
    msz = np.where(sz < mloc)[0]
    if len(msz) > 0:
        for d in msz:
            loc[:, d] = np.minimum(loc[:, d], sz[d])

    # Count for points where all coordinates are falling in a corresponding bins
    hasdata = np.all(loc > 0, axis=1)
    if AccumData is not None:
        if Fun is None:
            Fun = np.sum
        count = np.bincount(np.ravel_multi_index(loc[hasdata].T.astype(int), sz.astype(int)), 
                            weights=AccumData[hasdata], minlength=int(np.prod(sz)))
        count = count.reshape(sz.astype(int))
    else:
        count = np.histogramdd(X[hasdata], bins=edges)[0]

    return count, edges, mid, loc

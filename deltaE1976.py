import numpy as np

def deltaE1976(lab_ref, lab_est):
    """
    Compute the 1976 color difference (ΔE).

    Args:
        lab_ref (numpy.ndarray): Reference LABs.
        lab_est (numpy.ndarray): Estimated LABs.

    Returns:
        numpy.ndarray: Color difference.
    """
    dE = np.sqrt(np.sum((lab_ref - lab_est)**2, axis=1))
    return dE

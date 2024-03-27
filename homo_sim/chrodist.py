import numpy as np

def chrodist(rgb, Nbin):
    # CHIST computes normalised rg chromaticity 2D distribution
    # exclude saturated pixels
    ex_msk = rgb < 0.01
    ex_msk = ~np.logical_sum(ex_msk, axis=0)
    ex_msk = ex_msk & np.mean(rgb, axis=0) > 0.05
    C = np.array([[1,0,0],[0,1,0],[1,1,1]]) # base converse matrix
    pC = np.dot(C, rgb[:, ex_msk]) # chromaticity array
    hpC = pC / pC[2, :] # homogenous ones
    # quantitise the chromaticities
    hr = np.linspace(0, 1, Nbin+1)
    chist = np.histogram2d(hpC[0, :], hpC[1, :], bins=(hr, hr))[0]
    # make the probalities sum up to one
    chist = chist / np.max(chist)
    return chist


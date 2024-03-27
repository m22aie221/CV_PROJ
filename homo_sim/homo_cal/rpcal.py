"""
implementing the rpcal function along with its internal helper functions build_terms and cfun.
"""

import numpy as np

def rpcal(rgb, xyz, Morder=2):
    terms = build_terms(Morder)  # build terms

    Nterms = terms.shape[0]
    Npatch = rgb.shape[0]

    # Extend the poly terms
    pRGB = np.zeros((Npatch, Nterms))
    for ip in range(Npatch):
        pRGB[ip, :] = np.prod(rgb[ip, :] ** terms, axis=1)

    I = np.eye(Nterms) * 1e-7
    M_matrix = np.linalg.lstsq(pRGB.T @ pRGB + I.T @ I, pRGB.T @ xyz, rcond=None)[0]

    return {'matrix': M_matrix, 'terms': terms}


def build_terms(Mo):
    terms = np.zeros((0, 3))
    for nR in range(Mo, -1, -1):
        for nG in range(Mo - nR, -1, -1):
            nB = Mo - nR - nG
            terms = np.vstack((terms, [nR, nG, nB]))

    terms = terms / Mo
    return terms


def cfun(cRGB, cM, cterms):
    Nt = cterms.shape[0]
    Np = cRGB.shape[0]

    prgb = np.zeros((Np, Nt))
    for i in range(Np):
        prgb[i, :] = np.prod(cRGB[i, :] ** cterms, axis=1)

    cXYZ = np.dot(prgb, cM)  # Convert
    return cXYZ

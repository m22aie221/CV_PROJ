"""
the MATLAB functions dctmtx and dltmtx are not available directly in Python. You would need to implement them separately or find suitable replacements from libraries like NumPy or SciPy.
"""
import numpy as np

def uea_H_from_x_alsr(P, Q, csz, max_iter=30, solver='DCT', Morder=2, ind=None):
    if P.shape != Q.shape:
        raise ValueError('Input point sets are different sizes!')

    # size of valid intensities
    r, c = P.shape

    if ind is None:
        ind = np.arange(np.prod(csz))

    P = P.T
    Q = Q.T

    M = {}
    M['terms'] = build_terms(Morder)
    M['cfun'] = cfun

    Nterms = M['terms'].shape[1]
    Npatch = P.shape[1]

    # initialisation
    D = np.eye(c)
    N = P
    errs = np.inf * np.ones(max_iter + 1)  # error history

    gradient = create_basis(csz, solver)
    GRADbasis = np.zeros_like(gradient)
    num = gradient.shape[2]

    # extend the poly terms
    pP = np.zeros((Nterms, Npatch))
    for it in range(Nterms):
        ina = np.where(M['terms'][:, it] == 1)[0]
        if len(ina) == 0:
            pP[it, :] = np.prod(P ** np.tile(M['terms'][:, it], (1, Npatch)), axis=0)
        else:
            pP[it, :] = P[ina, :]

    # solve the homography using ALS
    n_it = 1
    while n_it - 1 < max_iter:
        n_it += 1  # increase number of iteration

        if solver in ['DCT', 'DLT']:
            imnew = N.T.reshape((*csz, 3))
            coef = gradbuilder(imnew, gradient, Q.T)

            for j in range(num):  # build shading gradient
                GRADbasis[:, :, j] = gradient[:, :, j] * coef[j]
            scs = np.sum(GRADbasis, axis=2)  # built shading
            D = np.diag(scs.flatten())

        tpP = np.dot(pP, D)
        H = np.dot(Q[:, ind], np.linalg.pinv(tpP[:, ind]))  # update H
        N = np.dot(H, pP)  # apply perspective transform

        errs[n_it] = np.mean((N.T @ D - Q) ** 2)  # mean square error

    if n_it > 2:
        pD = D

    M['matrix'] = H
    err = errs[n_it]

    return M, err, pD


def build_terms(Mo):
    terms = []
    for nR in range(Mo, -1, -1):
        for nG in range(Mo - nR, -1, -1):
            nB = Mo - nR - nG
            terms.append([nR, nG, nB])

    terms = np.array(terms) / Mo
    terms = terms.T
    tind = np.argmax(terms, axis=1)
    terms = terms[:, tind]
    
    return terms


def cfun(cRGB, cM, cterms):
    cRGB = cRGB.T
    Nt = cterms.shape[1]
    Np = cRGB.shape[1]

    prgb = np.zeros((Nt, Np))
    for tit in range(Nt):
        tina = np.where(cterms[:, tit] == 1)[0]
        if len(tina) == 0:
            prgb[tit, :] = np.prod(cRGB ** np.tile(cterms[:, tit], (1, Np)), axis=0)
        else:
            prgb[tit, :] = cRGB[tina, :]

    cXYZ = (cM @ prgb).T  # convert
    return cXYZ


def create_basis(sz, solver):
    switch = {
        'DCT': dctmtx,
        'DLT': dltmtx
    }
    f = switch.get(solver)

    P1 = f(sz[0])
    P2 = f(sz[1])

    order = 2
    basis = np.zeros((sz[0] * sz[1], np.sum(np.arange(1, order + 1))))
    sd = np.where(np.array(sz) == 1)[0]

    basis[:, 0] = np.kron(P1[0, :], P2[0, :])
    if len(sd) == 0 or sd[0] != 0:
        basis[:, 1] = np.kron(P1[1, :], P2[0, :])
    if len(sd) == 0 or sd[0] != 1:
        basis[:, 2] = np.kron(P1[0, :], P2[1, :])

    return basis


def fullimagegradienter(im, gradient):
    im_shaded = np.zeros_like(im)
    for lc in range(3):
        im_shaded[:, :, lc] = im[:, :, lc] * gradient
    return im_shaded


def gradbuilder(im, gradient, XYZ):
    Nb = gradient.shape[2]  # number of DCT basis
    A = np.zeros((np.prod(im.shape), Nb))

    for lj in range(Nb):
        im_shaded = fullimagegradienter(im, gradient[:, :, lj])
        A[:, lj] = im_shaded.flatten()

    A1 = A.T @ A
    coef = np.linalg.lstsq(A1 + np.eye(A.shape[1]) * np.mean(np.diag(A1)) * 1e-3, A.T @ XYZ.flatten(), rcond=None)[0]
    return coef

import numpy as np
from scipy.sparse import eye
class homo_solver():
    def uea_H_from_x_als(p1, p2, max_iter=50, tol=1e-20, k='lin'):
        Nch, Npx = p1.shape
        ind1 = np.sum((p1 > 0) & (p1 < np.inf), axis=0) == Nch
        ind2 = np.sum((p2 > 0) & (p2 < np.inf), axis=0) == Nch
        vind = ind1 & ind2
        kind = np.where(vind)[0]
        if p1.shape != p2.shape:
            raise ValueError('Input point sets are different sizes!')
        P = p1
        Q = p2
        N = P
        D = eye(Npx)
        errs = np.inf * np.ones(max_iter + 1)  # error history
        n_it = 1
        d_err = np.inf
        while n_it - 1 < max_iter and d_err > tol:
            n_it += 1
            D = homo_solver.SolveD1(N, Q)
            P_d = P @ D
            if k == 'lin':
                M = Q[:, vind] / P_d[:, vind]
            else:
                K = np.mean(np.diag(P_d @ P_d.T)) / 1e3
                M = ((P_d @ P_d.T + K * eye(Nch)) @ P_d @ Q.T).T
            N = M @ P
            NDiff = (N @ D - Q) ** 2
            errs[n_it] = np.mean(NDiff[:, vind])
            d_err = errs[n_it - 1] - errs[n_it]
        H = M
        err = errs[n_it]
        pD = D
        return H, err, pD

    def SolveD1(pp, qq):
        nCh, nPx = pp.shape
        p = pp
        q = qq
        Dr = np.arange(1, nPx * nCh + 1)  # row index
        Dc = np.tile(np.arange(1, nPx + 1), (nCh, 1)).flatten()  # column index
        A = np.zeros((nPx * nCh, nPx))
        A[Dr - 1, Dc - 1] = p.flatten()
        B = q.flatten()
        A1 = A.T @ A
        D = np.linalg.solve(A1, A.T @ B)
        D = np.diag(D)
        return D


"""
This code converts the MATLAB script to Python, using NumPy, Pandas, and some functions from scikit-image. Ensure you have these packages installed before running the Python script. Also, some functions, like alshomocal, ransachomocal_luv, lscal, alsRPcal, and rpcal, need to be defined or imported for this script to work correctly.
"""

import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from skimage.color import rgb2lab, rgb2luv, rgb2xyz
from skimage import img_as_float
from ransachomocal_luv import ransachomocal_luv
from HGluv2xyz import HGluv2xyz
from HGxyz2luv import HGxyz2luv
from uea_homocvt import uea_homocvt
from HGxyz2lab import HGxyz2lab

# Define utility functions

def deltaE1976(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))
'''
def XYZ2Lab(XYZ, wp):
    XYZ = XYZ / np.expand_dims(wp, axis=0)
    XYZ[XYZ > 0.008856] = XYZ[XYZ > 0.008856] ** (1 / 3)
    XYZ[XYZ <= 0.008856] = (7.787 * XYZ[XYZ <= 0.008856]) + (16 / 116)
    Lab = np.zeros_like(XYZ)
    Lab[..., 0] = (116 * XYZ[..., 1]) - 16
    Lab[..., 1] = 500 * (XYZ[..., 0] - XYZ[..., 1])
    Lab[..., 2] = 200 * (XYZ[..., 1] - XYZ[..., 2])
    return Lab


import numpy as np

def HGxyz2lab(xyz, wp=None):
    if wp is None:
        wp = np.array([0.950456, 1, 1.088754])

    X = xyz[:, 0] / wp[0]
    Y = xyz[:, 1] / wp[1]
    Z = xyz[:, 2] / wp[2]

    fX = f1(X)
    fY = f1(Y)
    fZ = f1(Z)

    lab = np.zeros_like(xyz)
    lab[:, 0] = 116 * fY - 16    # L*
    lab[:, 1] = 500 * (fX - fY)  # a*
    lab[:, 2] = 200 * (fY - fZ)  # b*

    return lab

def f1(Y):
    fY = np.real(Y**(1/3))
    i = (Y < 0.008856)
    fY[i] = Y[i] * (841/108) + (4/29)
    return fY

'''

def XYZ2Luv(XYZ, wp):
    Xr, Yr, Zr = wp
    eps = 0.008856
    kappa = 903.3
    u_r = (4 * Xr) / (Xr + (15 * Yr) + (3 * Zr))
    v_r = (9 * Yr) / (Xr + (15 * Yr) + (3 * Zr))
    Y = XYZ[..., 1] / Yr
    L = np.where(Y > eps, (116 * (Y ** (1 / 3))) - 16, kappa * Y)
    d = XYZ[..., 0] + (15 * XYZ[..., 1]) + (3 * XYZ[..., 2])
    u = (4 * XYZ[..., 0]) / d
    v = (9 * XYZ[..., 1]) / d
    u_prime = (13 * L * (u - u_r))
    v_prime = (13 * L * (v - v_r))
    return np.stack([L, u_prime, v_prime], axis=-1)

def XYZ2RGB(XYZ):
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    RGB = np.dot(XYZ, M.T)
    return np.clip(RGB, 0, 1)

def load_image(filename):
    return img_as_float(loadmat(filename)['cap']['sv'][0, 0])

'''
def HGxyz2lab(XYZ, wp):
    Lab = XYZ2Lab(XYZ, wp)
    return Lab
'''


def getAllFiles(dir_path):
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

def print_table(de, Method, de_str=None):
    disp_flag = de_str is not None
    de_mean = np.mean(de, axis=1)
    de_median = np.median(de, axis=1)
    de_95 = np.percentile(de, 95, axis=1)
    de_max = np.amax(de, axis=1)
    tab = pd.DataFrame({'Mean': de_mean, 'Median': de_median,
                        'pct95': de_95, 'Max': de_max}, 
                       index=Method)
    if disp_flag:
        print(de_str)
        print(tab)
    return tab

# Main script
dbpath = './data/HG_ColourChecker/'
#fmethod = [alshomocal, ransachomocal_luv, lscal, alsRPcal, rpcal]
#Method = ['ALS', 'ALS_RANSAC', 'LS', 'ALS_RP', 'RP']

fmethod = [ransachomocal_luv]
Method = ['ALS_RANSAC']

# Discover a list of images for conversion
fl = getAllFiles(os.path.join(dbpath, 'patch_real'))
fn = sorted(fl)

Npic = len(fn)
Npatch = 24
Nmethod = len(fmethod)

# Initialize arrays
de00_n = np.zeros((Npatch, Npic, Nmethod))
de76_n = np.zeros((Npatch, Npic, Nmethod))
deluv_n = np.zeros((Npatch, Npic, Nmethod))
dergb_n = np.zeros((Npatch, Npic, Nmethod))

de76_u = np.zeros((Npatch, Npic, Nmethod))
deluv_u = np.zeros((Npatch, Npic, Nmethod))
dergb_u = np.zeros((Npatch, Npic, Nmethod))

md = np.zeros((1, Npic, Nmethod))

for i, filename in enumerate(fn):
    cat = os.path.splitext(filename)[0].split('_')[0]
    cap = loadmat(os.path.join(dbpath, 'patch_real', filename))
    ref = loadmat(os.path.join(dbpath, f'ref_real-{cat}.mat'))

    ref = ref['ref']
    #xyz_std = ref['XYZ'] / ref['XYZ'][3, 1]

    xyz_std = ref['XYZ'][0] / ref['XYZ'][0][0][3, 1]
    
    xyz_values = xyz_std[0]
    xyz_std_3 = xyz_values[3, :]
    
    xyz_std = xyz_values
    
    lab_ref = HGxyz2lab(xyz_std, xyz_std_3)
    luv_ref = HGxyz2luv(xyz_std, xyz_std_3)
    rgb_ref = XYZ2RGB(xyz_std)

    fsv = cap['cap']['sv'][0, 0].reshape(-1, 3)
    fsv_uniform = cap['cap']['sv_uniform'][0, 0].reshape(-1, 3)

    for m in range(Nmethod):
        if Method[m] in ['ALS_RANSAC']:
            M_n = fmethod[m](fsv, xyz_std, xyz_std[3, :], fsv_uniform)
            M_u = fmethod[m](fsv_uniform, xyz_std, xyz_std[3, :], fsv_uniform)
        else:
            M_n = fmethod[m](fsv, xyz_std)
            M_u = fmethod[m](fsv_uniform, xyz_std)

        xyz_est_n = uea_homocvt(fsv_uniform, M_n)
        xyz_est_u = uea_homocvt(fsv_uniform, M_u)

        XYZ_est_n = xyz_est_n / xyz_est_n[3, 1]
        XYZ_est_u = xyz_est_u / xyz_est_u[3, 1]

        lab_est_n = HGxyz2lab(XYZ_est_n, xyz_std[3, :])
        de76_n[:, i, m] = deltaE1976(lab_ref, lab_est_n)

        lab_est_u = HGxyz2lab(XYZ_est_u, xyz_std[3, :])
        de76_u[:, i, m] = deltaE1976(lab_ref, lab_est_u)

        luv_est_n = HGxyz2luv(XYZ_est_n, xyz_std[3, :])
        deluv_n[:, i, m] = deltaE1976(luv_ref, luv_est_n)

        luv_est_u = HGxyz2luv(XYZ_est_u, xyz_std[3, :])
        deluv_u[:, i, m] = deltaE1976(luv_ref, luv_est_u)

        rgb_est_n = XYZ2RGB(XYZ_est_n)
        dergb_n[:, i, m] = deltaE1976(rgb_ref, rgb_est_n)

        rgb_est_u = XYZ2RGB(XYZ_est_u)
        dergb_u[:, i, m] = deltaE1976(rgb_ref, rgb_est_u)

# Print evaluation results (non-uniform shading)
trgb_n = print_table(dergb_n, Method, 'RGB (Non-Uniform)')
t76_n = print_table(de76_n, Method, 'DeltaE LAB 1976 (Non-Uniform)')
tluv_n = print_table(deluv_n, Method, 'DeltaE LUV (Non-Uniform)')

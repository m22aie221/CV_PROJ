# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:20:17 2024

@author: ranji
"""

import numpy as np

def HGxyz2lab(xyz, wp=None):
    if wp is None:
        wp = np.array([0.950456, 1, 1.088754])

    Y = xyz[:, 1] / wp[1]
    X = xyz[:, 0] / wp[0]
    Z = xyz[:, 2] / wp[2]
    
    fX = f(X)
    fY = f(Y)
    fZ = f(Z)
    
    lab = np.zeros_like(xyz)
    lab[:, 0] = 116 * fY - 16  # L*
    lab[:, 1] = 500 * (fX - fY)  # a*
    lab[:, 2] = 200 * (fY - fZ)  # b*
    
    return lab

def f(Y):
    fY = np.real(Y ** (1 / 3))
    i = Y < 0.008856
    fY[i] = Y[i] * (841 / 108) + (4 / 29)
    return fY

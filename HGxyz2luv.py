# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:23:11 2024

@author: ranji
"""

import numpy as np

def HGxyz2luv(xyz, white):
    if xyz.shape[1] != 3:
        print('xyz must be n by 3')
        return None

    luv = np.zeros_like(xyz)
    up = 4 * xyz[:, 0] / (xyz[:, 0] + 15 * xyz[:, 1] + 3 * xyz[:, 2])
    vp = 9 * xyz[:, 1] / (xyz[:, 0] + 15 * xyz[:, 1] + 3 * xyz[:, 2])
    upw = 4 * white[0] / (white[0] + 15 * white[1] + 3 * white[2])
    vpw = 9 * white[1] / (white[0] + 15 * white[1] + 3 * white[2])
    
    index = xyz[:, 1] / white[1] > 0.008856
    luv[:, 0] = luv[:, 0] + index * (116 * (xyz[:, 1] / white[1]) ** (1 / 3) - 16)
    luv[:, 0] = luv[:, 0] + (1 - index) * (903.3 * (xyz[:, 1] / white[1]))
    luv[:, 1] = 13 * luv[:, 0] * (up - upw)
    luv[:, 2] = 13 * luv[:, 0] * (vp - vpw)
    
    return luv, up, vp

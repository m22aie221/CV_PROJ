# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:27:11 2024

@author: ranji
"""

import numpy as np

def uea_homocvt(p, H, C=None):
    if C is None:
        C = np.eye(H.shape[0])
    
    pr = p.reshape(-1, H.shape[1]).T
    prC = np.dot(C, pr)
    qC = np.dot(H, prC)
    q = np.dot(np.linalg.inv(C), qC).T
    
    return q

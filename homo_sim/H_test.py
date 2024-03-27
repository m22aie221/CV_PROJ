import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys

from homo_solver.uea_H_from_x_als import homo_solver
import chrodist

def H_test(fn):
    # configuration
    dpath = './H_test_im/'
    
    # load query image
    rim = np.array(Image.open(dpath + str(fn) + '_i110.png')).astype(float) / 255
    pr = rim.reshape(-1, 3).T
    qim = np.array(Image.open(dpath + str(fn) + '_l6c1.png')).astype(float) / 255
    pq = qim.reshape(-1, 3).T
    
    # calculate H, M, pc11, pc12, pc2
    H, _, D = homo_solver.uea_H_from_x_als(pr, pq, 50)
    M = pq / pr
    pc11 = np.dot(H, pr) * D
    pc12 = np.dot(H, pr)
    pc2 = np.dot(M, pr)
    cim11 = pc11.T.reshape(qim.shape)
    cim12 = pc12.T.reshape(qim.shape)
    cim2 = pc2.T.reshape(qim.shape)
    
    # calculate ch1 and ch2
    ch1 = chrodist(pr, 128)
    ch2 = chrodist(pq, 128)
    
    # plot figures
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(rim)
    axs[0, 0].set_title('A')
    axs[0, 1].imshow(qim)
    axs[0, 1].set_title('B')
    axs[0, 2].imshow(cim11)
    axs[0, 2].set_title('A2B w shading')
    axs[1, 0].imshow(cim12)
    axs[1, 0].set_title('A2B w/o shading')
    axs[1, 1].imshow(cim2)
    axs[1, 1].set_title('A2B by Least-Squares')
    axs[1, 2].imshow(ch1)
    axs[1, 2].set_title('Chrodist')
    
    plt.show()

H_test(7)


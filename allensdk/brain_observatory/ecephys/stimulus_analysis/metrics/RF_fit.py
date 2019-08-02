# -*- coding: utf-8 -*-
"""
Created on Thu July 12 12:37:58 2018
@author: Xiaoxuan Jia
"""

# fit with 2D gaussian

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    #bnds = ((0, 50), (0, 18),(0,18),(0,2),(0,2))
    #p, success = optimize.minimize(errorfunction, params, bounds=bnds)
    return p, success

def signal_detection(X):
    """
    Xiaoxuan's invention of 1D signal detection.
    """
    X_median = np.nanmedian(X,axis=0)
    A = max(X) - min(X)
    e = X - X_median
    signal = A/(2*np.nanstd(e.flatten()))
    return signal


if __name__ == '__main__':

    #----------------------------------------------------
    data = np.random.randint(10, (10,10))

    params, success = fitgaussian(data)

    fit = gaussian(*params)

    plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    ax.imshow(data, cmap='inferno')
    ax.grid(False)

    ax.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    axx = plt.gca()
    (height, x, y, width_x, width_y) = params


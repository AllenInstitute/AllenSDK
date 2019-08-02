# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:09:39 2019
@author: Xiaoxuan Jia
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import scipy.optimize as opt


# -----------helper function
def findlevel(inwave, threshold, direction='both'):
    """
    adapted from Saskia
    """
    temp = inwave - threshold
    if (direction.find("up")+1):
        crossings = np.nonzero(np.ediff1d(np.sign(temp), to_begin=0)>0)
    elif (direction.find("down")+1):
        crossings = np.nonzero(np.ediff1d(np.sign(temp), to_begin=0)<0)
    else:
        crossings = np.nonzero(np.ediff1d(np.sign(temp), to_begin=0))
    return crossings[0][0]


def nanunique(x):
    """np.unique get rid of nan numbers."""
    temp = np.unique(x.astype(float))
    return temp[~np.isnan(temp)]

# -----------contrast tuning fit
def contrast_curve(x,b,c,d,e):
    """
     fit sigmoid function at log scale
     not good for fitting band pass
     - b: hill slope
     - c: min response
     - d: max response
     - e: EC50
    """
    return(c+(d-c)/(1+np.exp(b*(np.log(x)-np.log(e)))))

def gauss_function(x, a, x0, sigma):
    """
    fit gaussian function at log scale
    good for fitting band pass, not good at low pass or high pass
    """
    # fit band pass
    # fit gaussian on log transformed data
    return a*np.exp(-(np.log(x)-np.log(x0))**2/(2*sigma**2))

def logCon(x):
    """convert contrast to log10 scale"""
    return(-np.log10(1e-6*x))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_c50(x, y):
    """
    TODO: fix low pass and high pass
    """
    #x=cons
    #y=tuning[10,:]
    c_50_total = np.zeros(2)*np.NaN
    err = np.NaN
    try:
        fitCoefs, covMatrix = opt.curve_fit(contrast_curve, x, y, maxfev = 100000000)
        resids = y-contrast_curve(x,*fitCoefs)

        X = np.linspace(min(x)*0.9,max(x)*1.1,256)
        y_fit = contrast_curve(X,*fitCoefs)
        y_middle = (max(y_fit)-min(y_fit))/2+min(y_fit)
        idx_50=find_nearest(y_fit, y_middle)
        c_50=X[idx_50]
        y_50=y_fit[idx_50]
        # normalized MSE
        #MSE = np.sum((contrast_curve(x,*fitCoefs)-y)**2)
        # explained variance
        err = 1-(np.var(y-contrast_curve(x,*fitCoefs))/np.var(y))
        
        # define low pass or high pass
        c_50_total = np.zeros(2)*np.NaN
        if c_50>=x[np.argmax(y)]:
            # low pass: c50>x_max
            c_50_total[0]=c_50
        else:
            c_50_total[1]=c_50

        return c_50_total, err
    except:
        return c_50_total, err
        pass
    
def get_c50_gaussian(x, y):
    X = np.linspace(min(x)*0.9,max(x)*1.1,256)
    c_50 = np.zeros(2)*np.NaN
    err = np.NaN
    try:
        popt, pcov = opt.curve_fit(gauss_function, x, y, p0=[np.amax(y), x[np.argmax(y)], 1.], maxfev=200000)
        resids_gauss = tuning-gauss_function(x, *popt)
        
        y_fit = gauss_function(X, *popt)
        y_middle = (max(y_fit)-min(y_fit))/2+min(y_fit)
        
        #MSE = np.sum((gauss_function(x,*popt)-y)**2)
        err = 1-(np.var(y-gauss_function(x, *popt))/np.var(y))
        
        try:
            c_50[0] = X[findlevel(y_fit, y_middle, direction='up')]
            y_50 = y_fit[findlevel(y_fit, y_middle, direction='up')]
        except:
            pass
        try:
            c_50[1] =  X[findlevel(y_fit, y_middle, direction='down')]
            y_50 = y_fit[findlevel(y_fit, y_middle, direction='down')]
        except:
            pass
        return c_50, err
    except:
        return c_50, err
        pass
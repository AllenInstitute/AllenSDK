# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:16:20 2019
@author: Xiaoxuan Jia
"""
import numpy as np
from scipy import signal

#The periodogram can be efficiently computed using the fast Fourier transform (FFT). 
# There is a variety of methods, such as Welch and Blackman-Tukey methods, designed 
# to improve the performance using lag window functions either in the time domain or 
#in the correlation domain. In situations when the data length is short, to get a 
#smooth spectrum, we may increase the data length by padding zeros to the sequence.

def get_psd(sig, fs=1000., nperseg=256, plot=False,  method='default'):
    if method=='default':
        f, psd = signal.periodogram(sig, fs=fs, scaling='spectrum')
        
    if method=='welch':
        # nperseg: default is 256. Window size. Should be the 2^n value closest to signal length
        f, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
            
    if plot==True:
        plt.plot(f, psd)
        #plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.xlim([0,25])
    return f, psd

def get_complex_MI(psd, freq, tf=4):
    """
    ref: 2019 Zoccolan: Nonlinear Processing of Shape Information in Rat Lateral Extrastriate Cortex
    MI>3 indicates the signal 
    """
    MI = abs((psd[np.where(abs(freq-tf)==min(abs(freq-tf)))[0]] - 
              np.mean(psd))/np.sqrt(np.mean(psd**2)-np.mean(psd)**2))
    return MI

def main(data, fs, TF_pref, nperseg=256):
	"""
    data: time serie 
    fs: sampling rate of data in Hz
    TF_pref: temporal frequency hypothesis to test against
	"""
	freq, psd = get_psd(response, fs=fs, nperseg=nperseg, plot=True, method='welch')
	MI = get_complex_MI(psd, freq, tf=TF_pref)[0]
	return MI

import numpy as np
from scipy.ndimage import gaussian_filter

def get_onset_filter(binarized, threshold=0.1):
    """
    binarized: dark flase: units*repeats(75)*time(250)
    """    
    tmp = gaussian_filter(np.nanmean(binarized, axis=1), 4)*1000
    # subtract baseline for each cell
    tmp = tmp-np.tile(tmp[:,:30].mean(1), (tmp.shape[1],1)).T
    # normalize to the maximum of each neuron
    tmp = tmp/np.tile(np.max(tmp, axis=1), (tmp.shape[1],1)).T
    
    ONSET=[]
    for idx in range(tmp.shape[0]):
        onset = np.where(tmp[idx,:]>=threshold*np.max(tmp[idx,:]))[0][0]
        if onset>30 and onset<np.argmax(tmp[idx,:]) and onset<70:
            ONSET.append(onset)
        else:
            ONSET.append(np.NaN)
    ONSET=np.array(ONSET)
    return ONSET

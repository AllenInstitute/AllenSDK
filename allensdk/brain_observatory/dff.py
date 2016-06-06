import logging
import os, argparse
import matplotlib.pyplot as plt
import h5py 
import numpy as np

from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet

def movingmode_fast(x, kernelsize, y):
    """ Compute the windowed mode of an array.

    Parameters
    ----------
    x: np.ndarray
        Array to be analyzed
    kernelsize: int
        Size of the moving window
    y: np.ndarray
        Output array to store the results
    """

    histo = np.zeros(4096)
    halfsize = kernelsize/2
    for m in range (0,halfsize):
        histo[int(round(x[m]))] += 1

    mode = 1
    for n in range (2,1000):      #the range of possible mode value is < 1000
        if histo[n] > histo[mode]:
            mode = n
        
    # here initial mode is available
    for m in range (0,halfsize):
        q = int(round(x[halfsize+m]))
        histo[q] += 1
        if histo[q] > histo[mode]:
            mode = q
        y[m] = mode
    
    for m in range (halfsize,x.shape[0]-halfsize):
        p = int(round(x[m-halfsize]))
        histo[p] -= 1
        if p == mode:    #need to find possibly new mode value
            for n in range (2,1000):      #the range of possible mode value is < 1000
                if histo[n] > histo[mode]:
                    mode = n

        q = int(round(x[m+halfsize]))
        histo[q] += 1
        if histo[q] > histo[mode]:	
            mode = q

        y[m] = mode

    for m in range (x.shape[0]-halfsize,x.shape[0]):
        p = int(round(x[m-halfsize]))
        histo[p] -= 1
        if p == mode:    #need to find possibly new mode value
            for n in range (2,1000):      #the range of possible mode value is < 1000
                if histo[n] > histo[mode]:
                    mode = n

        y[m] = mode

    return 0

def movingaverage(x, kernelsize, y):
    """ Compute the windowed average of an array.

    Parameters
    ----------
    x: np.ndarray
        Array to be analyzed
    kernelsize: int
        Size of the moving window
    y: np.ndarray
        Output array to store the results
    """
    halfsize = kernelsize/2
    sumkernel = np.sum(x[0:halfsize])
    for m in range (0,halfsize):
        sumkernel = sumkernel + x[m+halfsize]
      	y[m] = sumkernel/(halfsize+m)

    sumkernel = np.sum(x[0:kernelsize])
    for m in range (halfsize,x.shape[0]-halfsize):
        sumkernel = sumkernel - x[m-halfsize] + x[m+halfsize]
      	y[m] = sumkernel/kernelsize

    for m in range (x.shape[0]-halfsize,x.shape[0]):
        sumkernel = sumkernel - x[m-halfsize]
      	y[m] = sumkernel/(halfsize-1+(x.shape[0]-m))

    return 0

def plot_onetrace(x1):
    """ Debug plotting function """
    q1 = 30000
    q2 = 60000
    q3 = 90000
    q4 = 120000

    yticks = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,10.0]
    plt.subplot(4,1,1)
    plt.plot(np.arange(0,q1),x1[0:q1],'g')
    plt.ylim(0, 10)
    plt.yticks(yticks)
    plt.xlabel('frames',fontsize=18)
    plt.ylabel('DF/F',fontsize=18)

    plt.subplot(4,1,2)
    plt.plot(np.arange(q1,q2),x1[q1:q2],'g')
    plt.ylim(0, 10)
    plt.yticks(yticks)
    plt.xlabel('frames',fontsize=18)
    plt.ylabel('DF/F',fontsize=18)

    plt.subplot(4,1,3)
    plt.plot(np.arange(q2,q3),x1[q2:q3],'g')
    plt.ylim(0, 10)
    plt.yticks(yticks)
    plt.xlabel('frames',fontsize=18)
    plt.ylabel('DF/F',fontsize=18)

    plt.subplot(4,1,4)
    plt.plot(np.arange(q3,x1.shape[0]),x1[q3:x1.shape[0]],'g')
    plt.ylim(0, 10)
    plt.yticks(yticks)
    plt.xlabel('frames',fontsize=18)
    plt.ylabel('DF/F',fontsize=18)

    return 0

def compute_dff(traces, save_plot_dir=None, mode_kernelsize=5400, mean_kernelsize=3000):
    """ Compute dF/F of a set of traces using a mean-shifted windowed mode operator. 
    The operation is basically:  

        T_mm = windowed_mean(windowed_mode(T))

        T_dff = (T - T_mm) / T_mm
    
    Parameters
    ----------
    traces: np.ndarray
       2D array of traces to be analyzed

    Returns
    -------
    np.ndarray with the same shape as the input array.
    """

    if save_plot_dir is not None and not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)

    logging.debug("trace matrix shape: %d %d" % (traces.shape[0], traces.shape[1]))

    modeline = np.zeros(traces.shape[1])
    modelineLP = np.zeros(traces.shape[1])
    dff = np.zeros((traces.shape[0], traces.shape[1]))

    logging.debug("computing df/f")

    for n in range(0,traces.shape[0]):
        movingmode_fast(traces[n,:], mode_kernelsize, modeline[:])
        movingaverage(modeline[:], mean_kernelsize, modelineLP[:])
	dff[n,:] = (traces[n,:] - modelineLP[:]) / modelineLP[:]

        logging.debug("finished trace %d/%d" % (n+1, traces.shape[0]))

        if save_plot_dir:
            fig = plt.figure(figsize=(150,40))
            plot_onetrace(dff[n,:])

            plt.title('ROI '+str(n)+' ', fontsize=18)
            fig.savefig(os.path.join(save_plot_dir,'dff_%d.png' % n), orientation='landscape')
            plt.close(fig)

    return dff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5")
    parser.add_argument("output_h5")
    parser.add_argument("--plot_dir")
    parser.add_argument("--log_level", default=logging.INFO)

    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    # read from "data"
    if args.input_h5.endswith("nwb"):
        timestamps, traces = BrainObservatoryNwbDataSet(args.input_h5).get_corrected_fluorescence_traces()
    else:
        input_h5 = h5py.File(args.input_h5, "r")
        traces = input_h5["data"].value
        input_h5.close()

    dff = compute_dff(traces, args.plot_dir)
    
    # write to "data"
    output_h5 = h5py.File(args.output_h5, "w")
    output_h5["data"] = dff
    output_h5.close()

if __name__ == "__main__": main()

import numpy as np
import logging
import h5py
import os
import logging

from scipy.optimize import nnls

class Demixer(object):
    """ Class for demixing signal between overlapping traces.
    NOTE: the demixed traces available for download use a `background_trace`
    from the motion-corrected movies, which are available upon request.

    Parameters
    ----------
    traces: np.ndarray
       2D array with rows = traces and columns = time points
       
    masks: np.ndarray
       num. cells x FOV height x FOV width array with ROI masks

    dt: float
       duration of one time step

    background_trace: np.ndarray
       1D array containing background signal from movie.
    """

    def __init__(self, traces, masks, dt=1./30., background_trace=None):
        self.traces = traces
        self.masks = masks
        self.background_trace = background_trace
        self.dt = dt

        N,T = traces.shape

        M,x,y = masks.shape

        if N != M:
            raise Exception("Cell numbers for traces and masks not equal: %d vs %d" % (N, M))

        self.N = N   # number of cell traces
        self.T = T   # time steps
        self.P = x*y   # number of pixels in mask
        self.x, self.y = (x,y)

        logging.warning("Demixed traces available for download via the AllenSDK "
                        "were run with a background trace computed using motion-corrected " 
                        "fluorescence videos.  These are available on request.")

        if background_trace is not None:
            if background_trace.shape[1] != T:
                raise Exception("Background trace length does not equal trace length: %d vs %d" % 
                                (len(background_trace), T))
            # stack the background trace on the bottom of the trace array
            self.traces = np.vstack([self.traces,background_trace])

            # add a filled mask to the end of the mask array for the background trace
            self.masks = np.vstack([self.masks,np.ones([x,y])[np.newaxis,:,:]])

            flat_masks = self.masks.reshape(N+1, x*y)
        else:
            flat_masks = self.masks.reshape(N, x * y)
        
        self.mask_overlap = np.dot(flat_masks, flat_masks.T)

    def save_demixed_traces(self, h5_path, h5_dataset="traces_demix"):
        """ Save the demixed traces to an HDF5 file. """

        demix_data = h5py.File(h5_path,'w')
        demix_data[h5_dataset] = self.get_demixed_traces()
        demix_data.close()

    def demix(self, positive_traces=False):
        """ Run the demixing algorithm and return the resulting traces.

        Parameters
        ----------
        positive_traces: bool
            If true, demix using non-negative least squares (slow).
            If false, use simple linear regression.

        Returns
        -------
        np.ndarray
            Array of demixed traces.  If the background trace was provided 
            on construction, it will be at the bottom of this array.
        """

        if positive_traces:
            return self.demix_non_negative_least_squares()
        else:
            return self.demix_regression()

    def demix_non_negative_least_squares(self):
        """ Demix the traces using non-negative least squares. """

        self.traces_demix = np.zeros(self.traces.shape)
        num_pixels_in_mask = np.sum(self.masks,axis=(1,2))
        F = self.traces.T*num_pixels_in_mask    # shape (T,N)

        # This could be sped up by avoiding the for loop, but would require reimplementing nnls
        for t in xrange(self.T):
            self.traces_demix[:,t] = nnls(self.mask_overlap, F[t])[0]

        return self.traces_demix

    def demix_regression(self):
        """ Demix the traces using linear regression. """

        num_pixels_in_mask = np.sum(self.masks,axis=(1,2))
        F = self.traces.T*num_pixels_in_mask
        F = F.T

        self.traces_demix =  np.dot(np.linalg.inv(self.mask_overlap), F)

        return self.traces_demix

    def get_demixed_traces(self):
        """ Return a traces x time steps array of demixed traces. If a background
        trace was provided, the results have the background trace signal removed.  
        """

        if self.background_trace is not None:
            return self.traces_demix[:-1]
        else:
            return self.traces_demix

    def get_traces_with_background(self):
        """ Return a traces x time steps array of demixed traces.  Adds background 
        trace signal back into results, if a background trace was provided.
        """

        if self.background_trace is not None:
            return self.get_demixed_traces() + self.traces_demix[-1]
        else:
            raise Exception("No background trace available.")


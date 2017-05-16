# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os
import argparse
import matplotlib.pyplot as plt
import h5py
import numpy as np

from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet


def movingmode_fast(x, kernelsize, y):
    """ Compute the windowed mode of an array.  A running mode is initialized
    with a histogram of values over the initial kernelsize/2 values.  The mode
    is then updated as the kernel moves by adding and subtracting values from
    the histogram.

    Parameters
    ----------
    x: np.ndarray
        Array to be analyzed
    kernelsize: int
        Size of the moving window
    y: np.ndarray
        Output array to store the results
    """

    # offset so that the trace is non-negative
    minval = min(x.min(), 0)
    if minval < 0:
        x = x - minval

    maxval = x.max()

    # compute a histogram of a half kernel
    halfsize = int(kernelsize / 2)
    histo = np.bincount(np.rint(x[:halfsize]).astype(
        np.uint32), minlength=int(maxval + 2))

    # find the mode of the first half kernel
    mode = np.argmax(histo)

    # here initial mode is available
    for m in range(0, halfsize):
        q = int(round(x[halfsize + m]))

        histo[q] += 1

        if histo[q] > histo[mode]:
            mode = q

        y[m] = mode

    for m in range(halfsize, x.shape[0] - halfsize):
        p = int(round(x[m - halfsize]))
        histo[p] -= 1

        # need to find possibly new mode value
        if p == mode:
            mode = np.argmax(histo)

        q = int(round(x[m + halfsize]))

        histo[q] += 1

        if histo[q] > histo[mode]:
            mode = q

        y[m] = mode

    for m in range(x.shape[0] - halfsize, x.shape[0]):
        p = int(round(x[m - halfsize]))
        histo[p] -= 1

        # need to find possibly new mode value
        if p == mode:
            mode = np.argmax(histo)

        y[m] = mode

    # undo the offset
    if minval < 0:
        y += minval

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

    halfsize = kernelsize / 2
    sumkernel = np.sum(x[0:halfsize])
    for m in range(0, halfsize):
        sumkernel = sumkernel + x[m + halfsize]
        y[m] = sumkernel / (halfsize + m)

    sumkernel = np.sum(x[0:kernelsize])
    for m in range(halfsize, x.shape[0] - halfsize):
        sumkernel = sumkernel - x[m - halfsize] + x[m + halfsize]
        y[m] = sumkernel / kernelsize

    for m in range(x.shape[0] - halfsize, x.shape[0]):
        sumkernel = sumkernel - x[m - halfsize]
        y[m] = sumkernel / (halfsize - 1 + (x.shape[0] - m))

    return 0


def plot_onetrace(dff, fc):
    """ Debug plotting function """
    qs = np.rint(np.linspace(0, len(dff), 5)).astype(int)

    dff_max = dff.max()
    dff_min = dff.min()
    fc_max = fc.max()
    fc_min = fc.min()

    for qi in range(len(qs) - 1):
        r = qs[qi], qs[qi + 1]

        frames = np.arange(r[0], r[1])
        ax = plt.subplot(len(qs), 1, qi + 1)
        ax.plot(frames, dff[r[0]:r[1]], 'g')
        ax.set_ylim(dff_min, dff_max)
        ax.set_xlim(r[0], r[1])
        ax.set_xlabel('frames', fontsize=18)
        ax.set_ylabel('DF/F', fontsize=18, color='g')

        ax = ax.twinx()
        ax.plot(frames, fc[r[0]:r[1]], 'b')
        ax.set_ylim(fc_min, fc_max)
        ax.set_xlim(r[0], r[1])
        ax.set_ylabel('FC', fontsize=18, color='b')

    return 0


def compute_dff(traces, save_plot_dir=None, mode_kernelsize=5400, mean_kernelsize=3000):
    """ Compute dF/F of a set of traces using a low-pass windowed-mode operator.
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

    if mode_kernelsize >= traces.shape[1]:
        raise Exception("Cannot compute dF/F: mode filter size (%d) longer than trace (%d)" %
                        (mode_kernelsize, traces.shape[1]))
    if mean_kernelsize >= traces.shape[1]:
        raise Exception("Cannot compute dF/F: mean filter size (%d) longer than trace (%d)" %
                        (mean_kernelsize, traces.shape[1]))

    if save_plot_dir is not None and not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)

    logging.debug("trace matrix shape: %d %d" %
                  (traces.shape[0], traces.shape[1]))

    modeline = np.zeros(traces.shape[1])
    modelineLP = np.zeros(traces.shape[1])
    dff = np.zeros((traces.shape[0], traces.shape[1]))

    logging.debug("computing df/f")

    for n in range(0, traces.shape[0]):
        if np.any(np.isnan(traces[n])):
            logging.warning("trace for roi %d contains NaNs, setting to NaN", n)
            dff[n,:] = np.nan
            continue

        movingmode_fast(traces[n, :], mode_kernelsize, modeline[:])
        movingaverage(modeline[:], mean_kernelsize, modelineLP[:])
        dff[n, :] = (traces[n, :] - modelineLP[:]) / modelineLP[:]

        logging.debug("finished trace %d/%d" % (n + 1, traces.shape[0]))

        if save_plot_dir:
            fig = plt.figure(figsize=(150, 40))
            plot_onetrace(dff[n, :], traces[n, :])

            plt.title('ROI ' + str(n) + ' ', fontsize=18)
            fig.savefig(os.path.join(save_plot_dir, 'dff_%d.png' %
                                     n), orientation='landscape')
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
        timestamps, traces = BrainObservatoryNwbDataSet(
            args.input_h5).get_corrected_fluorescence_traces()
    else:
        input_h5 = h5py.File(args.input_h5, "r")
        traces = input_h5["data"].value
        input_h5.close()

    dff = compute_dff(traces, args.plot_dir)

    # write to "data"
    output_h5 = h5py.File(args.output_h5, "w")
    output_h5["data"] = dff
    output_h5.close()

if __name__ == "__main__":
    main()

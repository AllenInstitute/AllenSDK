# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import scipy.sparse as sparse
import scipy.linalg as linalg
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import matplotlib.colors as colors
from allensdk.config.manifest import Manifest

def demix_time_dep_masks(raw_traces, stack, masks):
    '''

    :param raw_traces: extracted traces
    :param stack: movie (same length as traces)
    :param masks: binary roi masks
    :return: demixed traces
    '''
    N, T = raw_traces.shape
    _, x, y = masks.shape
    P = x * y

    if len(stack.shape) == 3:
        stack = stack.reshape(T, P)

    num_pixels_in_mask = np.sum(masks, axis=(1, 2))
    F = raw_traces.T * num_pixels_in_mask  # shape (T,N)
    F = F.T

    flat_masks = masks.reshape(N, P)
    flat_masks = sparse.csr_matrix(flat_masks)

    drop_frames = []
    demix_traces = np.zeros((N, T))

    for t in range(T):

        weighted_mask_sum = F[:, t]
        drop_test = (weighted_mask_sum == 0)

        if np.sum(drop_test == 0):
            norm_mat = sparse.diags(num_pixels_in_mask / weighted_mask_sum, offsets=0)
            stack_t = sparse.diags(stack[t], offsets=0)

            flat_weighted_masks = norm_mat.dot(flat_masks.dot(stack_t))

            overlap = flat_masks.dot(flat_weighted_masks.T).toarray()  # cast to dense numpy array for linear solver because solution is dense
            try:
                demix_traces[:, t] = linalg.solve(overlap, F[:, t])
            except linalg.LinAlgError as e:
                logging.warning("singular matrix, using least squares")
                x, _, _, _ = linalg.lstsq(overlap, F[:, t])
                demix_traces[:, t] = x

            drop_frames.append(False)

        else:
            drop_frames.append(True)

    return demix_traces, drop_frames

def plot_traces(raw_trace, demix_trace, roi_id, roi_ind, save_file):
    fig, ax = plt.subplots()

    ax.plot(raw_trace, label='Fluoresence')
    ax.plot(demix_trace, label='Demixed')
    ax.set_title("ROI ID(%d) index (%d)" % (roi_id, roi_ind))
    ax.legend()
    plt.savefig(save_file)
    plt.close(fig)

def find_zero_baselines(traces):
    means = traces.mean(axis=1)
    stds =  traces.std(axis=1)
    return np.where((means-stds) < 0)
    
def plot_negative_baselines(raw_traces, demix_traces, mask_array, roi_ids_mask, plot_dir, ext='png'):
    N, T = raw_traces.shape
    _, x, y = mask_array.shape

    logging.debug("finding negative baselines")
    neg_inds = find_negative_baselines(demix_traces)[0]
    
    overlap_inds = set()
    logging.debug("detected negative baselines: %s", str(neg_inds))
    for roi_ind in neg_inds:
        Manifest.safe_mkdir(plot_dir)

        save_file = os.path.join(plot_dir, str(roi_ids_mask[roi_ind]) + '_negative.' + ext)
        plot_traces(raw_traces[roi_ind], demix_traces[roi_ind], roi_ids_mask[roi_ind], roi_ind, save_file)

        ''' plot overlapping masks '''
        save_file = os.path.join(plot_dir, str(roi_ids_mask[roi_ind]) + '_negative_masks.' + ext)
        roi_overlap_inds = plot_overlap_masks_lengthOne(roi_ind, mask_array, save_file)

        overlap_inds.update(roi_overlap_inds)

    zero_inds = find_zero_baselines(demix_traces)[0]
    logging.debug("detected zero baselines: %s", str(zero_inds))
    overlap_inds.update(zero_inds)

    return list(overlap_inds)
        

    
        
def plot_negative_transients(raw_traces, demix_traces, valid_roi, mask_array, roi_ids_mask, plot_dir, ext='png'):

    N, T = raw_traces.shape
    _, x, y = mask_array.shape

    logging.debug("finding negative transients")
    trans_ind_list1 = [find_negative_transients_threshold(trace=demix_traces[n]) for n in range(N)]
    rois_with_trans1 = [i for i in range(N) if len(trans_ind_list1[i]) > 0]
    rois_with_trans = np.unique(rois_with_trans1)
    rois_with_trans = [r for r in rois_with_trans if len(trans_ind_list1[r][0]) > 0]

    logging.debug("plotting negative transients")

    flat_masks = mask_array.reshape(N, x*y)
    overlap = flat_masks.dot(flat_masks.T)
    overlap -= np.diag(np.diag(overlap))

    for roi_ind in rois_with_trans:

        ''' plot biggest negative transient of this roi '''
        trans_ind_list = trans_ind_list1[roi_ind]

        trans_ind_list = trans_ind_list[0]
        trans_list = []
        for i in trans_ind_list:
            if i > 100 and i < T - 100:
                trans_list.append(demix_traces[roi_ind, i - 100:i + 100])
            elif i > 100 and i >= T - 100:
                trans_list.append(demix_traces[roi_ind, i - 100:])
            else:
                trans_list.append(demix_traces[roi_ind, :i + 100])

        # trans_list = [demix_traces[roi_ind, i-100:i+100] for i in trans_ind_list if i > 100 and i < Nt]
        Ntrans = len(trans_list)
        biggest_trans = 0
        for i in range(1, Ntrans):
            if np.amin(trans_list[i]) < np.amin(trans_list[biggest_trans]):
                biggest_trans = i

        trans_ind = trans_ind_list[biggest_trans]

        # trans_ind_list = np.concatenate((trans_ind_list1[roi_ind][0], trans_ind_list2[roi_ind][0]))
        # trans_list_min = np.where(demix_traces[roi_ind, trans_ind_list] == min(demix_traces[roi_ind, trans_ind_list]))[0]

        if np.sum(overlap[roi_ind]) > 0:

            if valid_roi[roi_ind]:

                savefile = os.path.join(plot_dir, str(roi_ids_mask[roi_ind]) + '_transient_valid.' + ext)
                plot_transients(roi_ind, trans_ind, mask_array, raw_traces, demix_traces, savefile)

                ''' plot overlapping masks '''
                savefile = os.path.join(plot_dir, str(roi_ids_mask[roi_ind]) + '_masks_valid.' + ext)
                plot_overlap_masks_lengthOne(roi_ind, mask_array, savefile)
                # plot_overlap_masks(roi_ind, mask_test, savefile)
            else:
                savefile = os.path.join(plot_dir, str(roi_ids_mask[roi_ind]) + '_transient_invalid.' + ext)
                plot_transients(roi_ind, trans_ind, mask_array, raw_traces, demix_traces, savefile)

                ''' plot overlapping masks '''
                savefile = os.path.join(plot_dir, str(roi_ids_mask[roi_ind]) + '_masks_invalid.' + ext)
                plot_overlap_masks_lengthOne(roi_ind, mask_array, savefile)
                # plot_overlap_masks(roi_ind, mask_test, savefile)
                #
        else:
            continue

    return rois_with_trans


def rolling_window(trace, window=500):
    '''

    :param trace:
    :param window:
    :return:
    '''

    shape = trace.shape[:-1] + (trace.shape[-1] - window + 1, window)
    strides = trace.strides + (trace.strides[-1], )

    return np.lib.stride_tricks.as_strided(trace, shape=shape, strides=strides)


def find_negative_baselines(trace):
    means = trace.mean(axis=1)
    stds = trace.std(axis=1)
    return np.where((means+stds) < 0)

def find_negative_transients_threshold(trace, window=500, length=10, std_devs=3):
    trace = np.pad(trace, pad_width=(window-1, 0), mode='constant', constant_values=[np.mean(trace[:window])])
    rolling_mean = np.mean(rolling_window(trace, window), -1)
    rolling_std = np.std(rolling_window(trace, window), -1)

    below_thresh = (trace[window-1:] < rolling_mean - std_devs*rolling_std)
    below_thresh = np.pad(below_thresh, pad_width=(window-1, 0), mode='constant')
    trans_length = np.sum(rolling_window(below_thresh, length), -1)
    trans_length = trans_length[window-length:]

    trans_ind = np.where(trans_length == length)

    return trans_ind


def plot_overlap_masks_lengthOne(roi_ind, masks, savefile=None, weighted=False):

    masks = np.array(masks).astype(float)
    N, x, y = masks.shape
    if np.sum(masks[-1])  == x*y:
        masks = masks[:-1]
        N -= 1

    flat_masks = masks.reshape(N, x*y)
    masks_overlap = flat_masks.dot(flat_masks.T)

    ind_plot = np.where(masks_overlap[roi_ind, :] > 0)[0]  # rois (k) that roi_ind overlaps with
    for i in ind_plot:  # rois that overlap with each roi k
        ind_k = np.where(masks_overlap[i, :] > 0)[0]
        ind_plot = np.concatenate((ind_plot, ind_k))

    ind_plot = np.unique(ind_plot)
    ind_plot = np.concatenate(([roi_ind], ind_plot[ind_plot!=roi_ind]))

    plt.figure()
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    Ncol = len(color_list)
    for num, i in enumerate(ind_plot):
        mask_plot = masks[i]
        if not weighted:
            mask_plot = ((num % Ncol)+1)*np.ma.array(masks[i], mask=(masks[i] == 0))
            plt.imshow(mask_plot, clim=(1., Ncol+1), cmap=colors.ListedColormap(color_list), alpha=0.5, interpolation='nearest')
            # plt.imshow(mask_plot, clim=(1., len(ind_plot)), alpha=.5)

        elif weighted:
            mask_plot = np.ma.array(masks[i], mask=(masks[i] == 0))
            plt.imshow(mask_plot, cmap='gray_r', alpha=.5, interpolation='nearest')

        plt.text(np.mean(np.where(np.sum(mask_plot, axis=0))), np.mean(np.where(np.sum(mask_plot, axis=1))) ,str(i))

    mask_tot = np.sum(masks[ind_plot, :, :], axis=0)
    mask_x = np.sum(mask_tot, axis=0)
    mask_y = np.sum(mask_tot, axis=1)

    plt.xlim((np.amin(np.where(mask_x))-5, np.amax(np.where(mask_x))+5))
    plt.ylim((np.amin(np.where(mask_y))-5, np.amax(np.where(mask_y))+5))
    plt.title('Masks')

    if savefile is not None:
        plt.savefig(savefile)
    plt.close()

    return ind_plot


def plot_transients(roi_ind, t_trans, masks, traces, demix_traces, savefile):

    masks = np.array(masks).astype(float)
    N, x, y = masks.shape
    _, Nt = traces.shape

    flat_masks = masks.reshape(N, x*y)
    masks_overlap = flat_masks.dot(flat_masks.T)

    ind_plot = np.where(masks_overlap[roi_ind, :] > 0)[0]  # rois (k) that roi_ind overlaps with
    for i in ind_plot:  # rois that overlap with each roi k
        ind_k = np.where(masks_overlap[i, :] > 0)[0]
        ind_plot = np.concatenate((ind_plot, ind_k))

    ind_plot = np.unique(ind_plot)
    ind_plot = np.concatenate(([roi_ind], ind_plot[ind_plot!=roi_ind]))

    if t_trans > 150 and t_trans < Nt - 150:
        plot_t = range(t_trans - 150, t_trans + 150)
    elif t_trans > 150 and t_trans >= Nt - 150:
        plot_t = range(t_trans - 150, Nt)
    else:
        plot_t = range(0, t_trans + 150)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    Ncol = len(color_list)

    for num, i in enumerate(ind_plot):
        ax[0].plot(plot_t, traces[i, plot_t], label=str(i), color=color_list[(num % Ncol)])
        ax[1].plot(plot_t, demix_traces[i, plot_t], label=str(i), color=color_list[(num % Ncol)])

    ax[0].set_title('Raw')
    ax[0].set_ylabel('Fluorescence')
    ax[1].set_title('Demixed')
    ax[1].set_xlabel('Time')
    ax[0].legend(loc=0)

    plt.savefig(savefile)
    plt.close(fig)


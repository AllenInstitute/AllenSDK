from scipy.ndimage.filters import gaussian_filter
import numpy as np
import scipy.interpolate as spinterp
from .tools import memoize
import os
import warnings
from skimage.measure import block_reduce

def upsample_image(img, upsample=4):

    x = np.arange(16)
    y = np.arange(28)
    g = spinterp.interp2d(y, x, img, kind='linear')
    # offset = -(1 - .625)  # Offset so that linear interpolation doesnt get biased
    offset=0
    ZZ_on = g(offset + np.arange(0, 28, 1. / upsample), offset + np.arange(0, 16, 1. / upsample))

    return ZZ_on


def convolve(img, upsample=4, sigma=4):
    '''
    2D Gaussian convolution
    '''

    if img.sum() == 0:
        return img

    img_pad = np.zeros((3 * img.shape[0], 3 * img.shape[1]))
    img_pad[img.shape[0]:2 * img.shape[0], img.shape[1]:2 * img.shape[1]] = img

    x = np.arange(3 * 16)
    y = np.arange(3 * 28)
    g = spinterp.interp2d(y, x, img_pad, kind='linear')
    offset = -(1 - .625)  # Offset so that linear interpolation doesnt get biased
    assert upsample == 4  # Offset needs to be computed for each upsample...
    ZZ_on = g(offset + np.arange(0, 28 * 3, 1. / upsample), offset + np.arange(0, 16 * 3, 1. / upsample))
    ZZ_on_f = gaussian_filter(ZZ_on, float(sigma), mode='constant')

    # For debugging:
    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # ax[0].imshow(ZZ_on, interpolation='none')
    # ax[1].imshow(ZZ_on_f, interpolation='none')
    # plt.show()

    z_on_new = block_reduce(ZZ_on_f, (upsample, upsample))
    z_on_new = z_on_new / z_on_new.sum() * img.sum()
    z_on_new = z_on_new[img.shape[0]:2 * img.shape[0], img.shape[1]:2 * img.shape[1]]

    return z_on_new

def plot_fields(data, axes=None, show=True, clim=(0, 1), colorbar=True):

    import matplotlib.pyplot as plt
    from matplotlib import ticker

    data = np.array(data)
    if axes is None:
        _, axes = plt.subplots(1, 2)

    axes[0].imshow(data[:16 * 28].reshape(16, 28), clim=clim, interpolation='none', origin='lower')
    img = axes[1].imshow(data[16 * 28:].reshape(16, 28), clim=clim, interpolation='none', origin='lower')
    if colorbar == True:
        cb = axes[0].figure.colorbar(img, ax=axes[1])
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

    if show == True:
        plt.show()

@memoize
def get_A(data):

    import warnings
    warnings.warn('FOR DEBUG')

    try:

        warnings.warn('A loaded')
        return np.load('./A_tmp.npy')
    except:


        stimulus_table = data.get_stimulus_table('locally_sparse_noise')
        stimulus_template = data.get_stimulus_template('locally_sparse_noise')[stimulus_table['frame'].values, :,:]

        A = np.zeros((2*16*28, stimulus_template.shape[0]))
        for fi in range(stimulus_template.shape[0]):
            A[:16*28, fi] = (stimulus_template[fi,:,:].flatten() > 127).astype(float)
            A[16*28:, fi] = (stimulus_template[fi, :, :].flatten() < 127).astype(float)

        assert A[:,100].sum() == 12
        assert A[:,200].sum() == 10
        assert A.shape == (896,8880)

        np.save('./A_tmp.npy', A)


    return A

@memoize
def get_A_blur(data, debug=False):

    import warnings
    warnings.warn('FOR DEBUG')

    try:
        warnings.warn('A_blur loaded')
        return np.load('./A_tmp.npy')

    except:

        if debug == True:
            warnings.warn("DEBUG MODE: Loading A_blur")
            A_blur_location = '/data/mat/nicholasc/brain_observatory_analysis/receptivefield/receptivefield/allensdk_tools/allensdk_tools_cache/A_blur.h5'
            A_blur = get_cache_array_sparse_h5_reader_writer()['reader'](A_blur_location)
            return A_blur
        else:
            A = get_A(data).copy()

            for fi in range(A.shape[1]):
                A[:16*28,fi] = convolve(A[:16 * 28, fi].reshape(16, 28)).flatten()
                A[16*28:,fi] = convolve(A[16 * 28:, fi].reshape(16, 28)).flatten()

        np.save('./A_blur_tmp.npy', A)

    return A

def get_shuffle_matrix(data, number_of_events, number_of_shuffles=5000, response_detection_error_std_dev=.1, debug=False):

    A = get_A_blur(data, debug=debug)

    shuffle_data = np.zeros((2*16*28, number_of_shuffles))
    for ii in range(number_of_shuffles):

        size = number_of_events + int(np.round(response_detection_error_std_dev*number_of_events*np.random.randn()))
        shuffled_event_inds = np.random.choice(range(8880), size=size, replace=False)
        b_tmp = np.zeros(8880)
        b_tmp[shuffled_event_inds] = 1
        shuffle_data[:, ii] = A.dot(b_tmp)/float(size)

    return shuffle_data

def get_sparse_noise_epoch_mask_list(st, number_of_acquisition_frames, threshold=7):

    delta = (st.start.values[1:] - st.end.values[:-1])
    cut_inds = np.where(delta > threshold)[0] + 1

    epoch_mask_list = []

    if len(cut_inds) > 2:
        warnings.warn('more than 2 epochs cut')
        print '    ', len(delta), cut_inds

    for ii in range(len(cut_inds)+1):

        if ii == 0:
            first_ind = st.iloc[0].start
        else:
            first_ind = st.iloc[cut_inds[ii-1]].start

        if ii == len(cut_inds):
            last_ind_inclusive = st.iloc[-1].end
        else:
            last_ind_inclusive = st.iloc[cut_inds[ii]-1].end

        epoch_mask_list.append((first_ind,last_ind_inclusive))

    return epoch_mask_list

def smooth(x,window_len=11,window='hanning', mode='valid'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode=mode)
    return y


def get_components(receptive_field_data):

    candidate_pixel_list = np.where(receptive_field_data.flatten()==True)[0]
    pixel_coord_dict = dict((px, (px/28, (px - 28 * (px/28)), px% (16 * 28) == px)) for px in candidate_pixel_list)

    component_list = []

    for curr_pixel in candidate_pixel_list:

        curr_x, curr_y, curr_frame = pixel_coord_dict[curr_pixel]

        component_list.append([curr_pixel])
        dist_to_component_dict = {}
        for ii, curr_component in enumerate(component_list):
            dist_to_component_dict[ii] = np.inf
            for other_pixel in curr_component:

                other_x, other_y, other_frame = pixel_coord_dict[other_pixel]

                if other_frame == curr_frame:
                    x_dist = np.abs(curr_x - other_x)
                    y_dist = np.abs(curr_y - other_y)
                    curr_dist = max(x_dist, y_dist)
                    if curr_dist <  dist_to_component_dict[ii]:
                        dist_to_component_dict[ii] = curr_dist

        # Merge all components with a distance leq 1 to current pixel
        new_component_list = []
        tmp = []
        for ii, curr_component in enumerate(component_list):
            if dist_to_component_dict[ii] <= 1:
                tmp += curr_component
            else:
                new_component_list.append(curr_component)

        new_component_list.append(tmp)
        component_list = new_component_list


    if len(component_list) == 0:
        return np.zeros((1, receptive_field_data.shape[0], receptive_field_data.shape[1])), len(component_list)
    elif len(component_list) == 1:
        return_array = np.zeros((1,receptive_field_data.shape[0], receptive_field_data.shape[1]))
    else:
        return_array = np.zeros((len(component_list), receptive_field_data.shape[0], receptive_field_data.shape[1]))

    for ii, component in enumerate(component_list):
        curr_component_mask = np.zeros_like(receptive_field_data, dtype=np.bool).flatten()
        curr_component_mask[component] = True
        return_array[ii,:,:] = curr_component_mask.reshape(receptive_field_data.shape)



    return return_array, len(component_list)


if __name__ == "__main__":

    csid = 517526760

    from receptivefield.core import get_receptive_field_data_dict_with_postprocessing, write_receptive_field_data_dict_to_h5, print_summary, read_receptive_field_data_dict_from_h5

    receptive_field_data_dict = get_receptive_field_data_dict_with_postprocessing(csid=csid, alpha=.05, debug=True)
    # receptive_field_data_dict = read_receptive_field_data_dict_from_h5('tmp.h5', path=str(csid))
    # write_receptive_field_data_dict_to_h5(receptive_field_data_dict, 'tmp.h5', prefix=str(csid))
    print_summary(receptive_field_data_dict)
    # plot_receptive_field_data(receptive_field_data_dict)

    # import time
    #
    # number_of_events = 100
    #
    # t0 = time.time()
    # A = get_shuffle_matrix(number_of_events, debug=True)
    # print time.time() - t0
    #
    # t0 = time.time()
    # A = get_shuffle_matrix(number_of_events, debug=True)
    # print time.time() - t0
    #
    # t0 = time.time()
    # A = get_shuffle_matrix(number_of_events, debug=True)
    # print time.time() - t0
    #
    # print A.shape
# Copyright 2017 Allen Institute for Brain Science
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

import numpy as np
import scipy.interpolate as si
import scipy.ndimage.filters as filt
import scipy.stats as stats

def chi_square_binary(events, LSN_template):
    # note: can only be applied to binary events for trial responses
    #
    # *****INPUT*****
    # events: 2D numpy bool with shape (num_trials,num_cells) for presence
    #   or absence of response on a given trial
    # LSN_template: 3D numpy int8 with shape (num_trials,num_y_pixels,num_x_pixels)
    #   for luminance at each pixel location
    #
    # *****OUTPUT*****
    # chi_square_grid_NLL: 3D numpy float with shape (num_cells,num_y_pixels,num_x_pixels)
    #   that gives the negative log-likelihood that a receptive field is contained
    #   within a 7x7 pixel mask centered on a given pixel location as measured
    #   by a chi-square test for the responses among the pixels (both on and off)
    #   that fall within the mask.

    num_trials = np.shape(events)[0]
    num_cells = np.shape(events)[1]
    num_y = np.shape(LSN_template)[1]
    num_x = np.shape(LSN_template)[2]

    # for each pixel location, get a mask that is centered on that location
    #    disc_masks has shape (num_y,num_x,num_y,num_x)
    disc_masks = get_disc_masks(LSN_template)

    # determine which trials each pixel is active (i.e not gray),
    #   broken up by ON and OFF pixels.
    #   trial_matrix has shape (num_y,num_x,2,num_trials)
    trial_matrix = build_trial_matrix(LSN_template, num_trials)

    # get the total number of trials each pixel is active (i.e. not gray)
    #   trials_per_pixel has shape (num_y,num_x,2)
    trials_per_pixel = np.sum(trial_matrix, axis=3)

    # get the sum of the number of events across all trials that each pixel is active
    #   events_per_pixel has shape (num_cells,num_y,num_x,2)
    events_per_pixel = get_events_per_pixel(events, trial_matrix)

    # smooth stimulus-triggered average spatially with a gaussian
    for n in range(num_cells):
        for on_off in range(2):
            events_per_pixel[n, :, :, on_off] = smooth_STA(events_per_pixel[n, :, :, on_off])

    # calculate the p_value for each exclusion region
    chi_square_grid = np.zeros((num_cells, num_y, num_x))
    for y in range(num_y):
        for x in range(num_x):
            exclusion_mask = np.ones((num_y, num_x, 2)) * disc_masks[y, x, :, :].reshape(num_y, num_x, 1)
            p_vals, __ = chi_square_within_mask(exclusion_mask, events_per_pixel, trials_per_pixel)
            chi_square_grid[:, y, x] = p_vals

    # convert p-values to negative log-likelihood
    # chi_square_grid_NLL = pvalue_to_NLL(chi_square_grid)

    return chi_square_grid



def get_peak_significance(chi_square_grid_NLL,
                          LSN_template,
                          alpha=0.05):
    # ****INPUT*****
    # chi_square_grid_NLL: result of chi_square_binary(events,LSN_template)
    # LSN_template: 3D numpy int8 with shape (num_trials,num_y_pixels,num_x_pixels)
    #   for luminance at each pixel location
    #
    # *****OUTPUT*****
    # significant_cells: 1D numpy bool with shape (num_cells,) that indicates
    #   whether or not each cell has a location with a significant chance of a
    #   true receptive field
    # best_exclusion_region_list: list of 2D numpy bool with len (num_cells). For each cell,
    #   the array is a mask for the best pixel, or all false

    num_cells = np.shape(chi_square_grid_NLL)[0]
    num_y = np.shape(chi_square_grid_NLL)[1]
    num_x = np.shape(chi_square_grid_NLL)[2]

    chi_square_grid = NLL_to_pvalue(chi_square_grid_NLL)

    # get the average size of all masks in units of number of pixels
    disc_masks = get_disc_masks(LSN_template)
    pixels_per_mask_per_pixel = np.sum(disc_masks, axis=(2, 3)).astype(float)

    # find the smallest p-value and determine if it's significant
    significant_cells = np.zeros((num_cells)).astype(bool)
    best_p = np.zeros((num_cells))
    p_value_correction_factor_per_pixel = (1.0 * num_y * num_x / pixels_per_mask_per_pixel)

    best_exclusion_region_list = []
    corrected_p_value_array_list = []
    for n in range(num_cells):

        # Sidak correction:
        p_value_corrected_per_pixel = 1-np.power((1-chi_square_grid[n, :,:]), p_value_correction_factor_per_pixel)
        corrected_p_value_array_list.append(p_value_corrected_per_pixel)

        y, x = np.unravel_index(p_value_corrected_per_pixel.argmin(), (num_y, num_x))

        # if more than one p-value that maxes out, use the median location
        if np.sum(p_value_corrected_per_pixel == 0.0) > 1:

            y, x = np.unravel_index(np.argwhere(p_value_corrected_per_pixel.flatten() == 0.0)[:, 0], (num_y, num_x))

            med_x = np.median(x)
            med_y = np.median(y)
            center_x = x[0]
            center_y = y[0]
            for i in range(len(x)):
                dx = x[i] - med_x
                dy = y[i] - med_y
                dc_x = center_x - med_x
                dc_y = center_y - med_y
                if np.sqrt(dx ** 2 + dy ** 2) < np.sqrt(dc_x ** 2 + dc_y ** 2):
                    center_x = x[i]
                    center_y = y[i]
            x = center_x
            y = center_y

        best_p[n] = p_value_corrected_per_pixel[y,x]
        if best_p[n] < alpha:
            significant_cells[n] = True
            best_exclusion_region_list.append(disc_masks[y, x, :,:].astype(np.bool))
        else:
            best_exclusion_region_list.append(np.zeros((disc_masks.shape[0], disc_masks.shape[1]), dtype=np.bool))

    return significant_cells, best_p, corrected_p_value_array_list, best_exclusion_region_list


def pvalue_to_NLL(p_values,
                  max_NLL=10.0):
    return np.where(p_values == 0.0, max_NLL, -np.log10(p_values))


def NLL_to_pvalue(NLLs,
                  log_base=10.0):
    return (log_base ** (-NLLs))


def get_events_per_pixel(responses_np,
                         trial_matrix):
    num_cells = np.shape(responses_np)[1]
    num_y = np.shape(trial_matrix)[0]
    num_x = np.shape(trial_matrix)[1]

    events_per_pixel = np.zeros((num_cells, num_y, num_x, 2))
    for y in range(num_y):
        for x in range(num_x):
            for on_off in range(2):
                frames = np.argwhere(trial_matrix[y, x, on_off, :])[:, 0]
                events_per_pixel[:, y, x, on_off] = np.sum(responses_np[frames, :], axis=0)

    return events_per_pixel


def smooth_STA(STA,
               gauss_std=0.75):

    deg_per_pnt = 64/STA.shape[0]
    STA_interpolated = interpolate_RF(STA, deg_per_pnt)
    STA_interpolated_smoothed = filt.gaussian_filter(STA_interpolated, gauss_std)
    STA_smoothed = deinterpolate_RF(STA_interpolated_smoothed, STA.shape[1], STA.shape[0], deg_per_pnt)

    return STA_smoothed


def interpolate_RF(rf_map,
                   deg_per_pnt):
    x_pnts = np.shape(rf_map)[1]
    y_pnts = np.shape(rf_map)[0]

    x_coor = np.arange(-(x_pnts - 1) * deg_per_pnt / 2, (x_pnts + 1) * deg_per_pnt / 2, deg_per_pnt)
    y_coor = np.arange(-(y_pnts - 1) * deg_per_pnt / 2, (y_pnts + 1) * deg_per_pnt / 2, deg_per_pnt)

    x_interpolated = np.arange(-(x_pnts - 1) * deg_per_pnt / 2, deg_per_pnt / 2 + (x_pnts / 2 - 1) * deg_per_pnt + 1, 1)
    y_interpolated = np.arange(-(y_pnts - 1) * deg_per_pnt / 2, deg_per_pnt / 2 + (y_pnts / 2 - 1) * deg_per_pnt + 1, 1)

    interpolated = si.interp2d(x_coor, y_coor, rf_map)
    interpolated = interpolated(x_interpolated, y_interpolated)

    return interpolated


def deinterpolate_RF(rf_map, x_pnts, y_pnts, deg_per_pnt):

    # x_pnts = 28
    # y_pnts = 16

    x_interpolated = np.arange(-(x_pnts - 1) * deg_per_pnt / 2, deg_per_pnt / 2 + (x_pnts / 2 - 1) * deg_per_pnt + 1, 1)
    y_interpolated = np.arange(-(y_pnts - 1) * deg_per_pnt / 2, deg_per_pnt / 2 + (y_pnts / 2 - 1) * deg_per_pnt + 1, 1)

    x_deinterpolate = np.arange(0, len(x_interpolated), deg_per_pnt)
    y_deinterpolate = np.arange(0, len(y_interpolated), deg_per_pnt)

    sampled_y = rf_map[y_deinterpolate, :]
    sampled_yx = sampled_y[:, x_deinterpolate]

    return sampled_yx


def chi_square_within_mask(exclusion_mask,
                           events_per_pixel,
                           trials_per_pixel):
    num_y = np.shape(exclusion_mask)[0]
    num_x = np.shape(exclusion_mask)[1]
    num_cells = np.shape(events_per_pixel)[0]

    # d.f. is number of pixels in mask minus one
    degrees_of_freedom = int(np.sum(exclusion_mask)) - 1

    # calculate expected number of events per pixel
    #   expected_by_pixel has shape (num_cells,num_y,num_x,2)
    num_trials_by_pixel = exclusion_mask * trials_per_pixel  # shape is (num_y,num_x,2)
    total_trials_in_mask = np.sum(num_trials_by_pixel).astype(float)
    events_by_pixel_in_mask = exclusion_mask.reshape(1, num_y, num_x, 2) * events_per_pixel
    total_events_in_mask = np.sum(events_by_pixel_in_mask, axis=(1, 2, 3)).astype(float)
    expected_per_trial = total_events_in_mask / total_trials_in_mask  # shape is (num_cells,)
    expected_by_pixel = num_trials_by_pixel.reshape(1, num_y, num_x, 2) * expected_per_trial.reshape(num_cells, 1, 1, 1)

    # calculate observed number of events per pixel
    #   observed_by_pixel has shape (num_cells,num_y,num_x,2)
    observed_by_pixel = (events_per_pixel * exclusion_mask.reshape(1, num_y, num_x, 2)).astype(float)

    # calculate test statistic given observed and expected
    residual_by_pixel = observed_by_pixel - expected_by_pixel
    chi = (residual_by_pixel ** 2) / expected_by_pixel
    chi_sum = np.nansum(chi, axis=(1, 2, 3))

    # get p-value given test statistic and degrees of freedom
    p_vals = 1.0 - stats.chi2.cdf(chi_sum, degrees_of_freedom)

    return p_vals, chi


def build_trial_matrix(LSN_template,
                       num_trials):
    num_y = np.shape(LSN_template)[1]
    num_x = np.shape(LSN_template)[2]
    on_off_luminance = [255, 0]

    trial_mat = np.zeros((num_y, num_x, 2, num_trials), dtype=bool)
    for y in range(num_y):
        for x in range(num_x):
            for on_off in range(2):
                frame = np.argwhere(LSN_template[:num_trials, y, x] == on_off_luminance[on_off])[:, 0]
                trial_mat[y, x, on_off, frame] = True

    return trial_mat


def get_disc_masks(LSN_template,
                   radius=3):
    on_luminance = 255
    off_luminance = 0
    num_y = np.shape(LSN_template)[1]
    num_x = np.shape(LSN_template)[2]

    # convert template to true on trials a pixel is not gray and false when gray
    LSN_binary = np.where(LSN_template == off_luminance, 1, LSN_template)
    LSN_binary = np.where(LSN_binary == on_luminance, 1, LSN_binary)
    LSN_binary = np.where(LSN_binary == 1, 1.0, 0.0)

    # get number of trials each pixel is not gray
    on_trials = np.sum(LSN_binary, axis=0).astype(float)  # shape is (num_y,num_x)

    masks = np.zeros((num_y, num_x, num_y, num_x))
    for y in range(num_y):
        for x in range(num_x):
            trials_not_gray = np.argwhere(LSN_binary[:, y, x] > 0)[:, 0]
            raw_mask = np.divide(np.sum(LSN_binary[trials_not_gray, :, :], axis=0), on_trials)

            center_y, center_x = np.unravel_index(raw_mask.argmax(), (num_y, num_x))

            # include center pixel in mask
            raw_mask[center_y, center_x] = 0.0

            x_max = center_x + radius + 1
            if x_max > num_x:
                x_max = num_x
            x_min = center_x - radius
            if x_min < 0:
                x_min = 0
            y_max = center_y + radius + 1
            if y_max > num_y:
                y_max = num_y
            y_min = center_y - radius
            if y_min < 0:
                y_min = 0

            # don't include far away pixels that just happen
            # to not have any trials in common with center pixel
            clean_mask = np.ones(np.shape(raw_mask))
            clean_mask[y_min:y_max, x_min:x_max] = raw_mask[y_min:y_max, x_min:x_max]

            masks[y, x, :, :] = clean_mask

    masks = np.where(masks > 0, 0.0, 1.0)

    return masks


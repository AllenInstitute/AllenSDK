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

from .utilities import smooth
import numpy as np
import scipy.stats as sps

def detect_events(data, cell_index, stimulus, debug_plots=False):



    stimulus_table = data.get_stimulus_table(stimulus)
    dff_trace = data.get_dff_traces()[1][cell_index, :]



    k_min = 0
    k_max = 10
    delta = 3

    dff_trace = smooth(dff_trace, 5)



    var_dict = {}
    debug_dict = {}
    for ii, fi in enumerate(stimulus_table['start'].values):

        if ii > 0 and stimulus_table.iloc[ii].start == stimulus_table.iloc[ii-1].end:
            offset = 1
        else:
            offset = 0

        if fi + k_min >= 0 and fi + k_max <= len(dff_trace):
            trace = dff_trace[fi + k_min+1+offset:fi + k_max+1+offset]

            xx = (trace - trace[0])[delta] - (trace - trace[0])[0]
            yy = max((trace - trace[0])[delta + 2] - (trace - trace[0])[0 + 2],
                     (trace - trace[0])[delta + 3] - (trace - trace[0])[0 + 3],
                     (trace - trace[0])[delta + 4] - (trace - trace[0])[0 + 4])

            var_dict[ii] = (trace[0], trace[-1], xx, yy)
            debug_dict[fi + k_min+1+offset] = (ii, trace)

    xx_list, yy_list = [], []
    for _, _, xx, yy in var_dict.itervalues():
        xx_list.append(xx)
        yy_list.append(yy)

    mu_x = np.median(xx_list)
    mu_y = np.median(yy_list)

    xx_centered = np.array(xx_list)-mu_x
    yy_centered = np.array(yy_list)-mu_y

    std_factor = 1
    std_x = 1./std_factor*np.percentile(np.abs(xx_centered), [100*(1-2*(1-sps.norm.cdf(std_factor)))])
    std_y = 1./std_factor*np.percentile(np.abs(yy_centered), [100*(1-2*(1-sps.norm.cdf(std_factor)))])

    curr_inds = []
    allowed_sigma = 4
    for ii, (xi, yi) in enumerate(zip(xx_centered, yy_centered)):
        if np.sqrt(((xi)/std_x)**2+((yi)/std_y)**2) < allowed_sigma:
            curr_inds.append(True)
        else:
            curr_inds.append(False)

    curr_inds = np.array(curr_inds)
    data_x = xx_centered[curr_inds]
    data_y = yy_centered[curr_inds]
    Cov = np.cov(data_x, data_y)
    Cov_Factor = np.linalg.cholesky(Cov)
    Cov_Factor_Inv = np.linalg.inv(Cov_Factor)

    #===================================================================================================================

    noise_threshold = max(allowed_sigma * std_x + mu_x, allowed_sigma * std_y + mu_y)
    mu_array = np.array([mu_x, mu_y])
    yes_set, no_set = set(), set()
    for ii, (t0, tf, xx, yy) in var_dict.iteritems():


        xi_z, yi_z = Cov_Factor_Inv.dot((np.array([xx,yy]) - mu_array))

        # Conditions in order:
        # 1) Outside noise blob
        # 2) Minimum change in df/f
        # 3) Change evoked by this trial, not previous
        # 4) At end of trace, ended up outside of noise floor

        if np.sqrt(xi_z**2 + yi_z**2) > 4 and yy > .05 and xx < yy and tf > noise_threshold/2:
            yes_set.add(ii)
        else:
            no_set.add(ii)



    assert len(var_dict) == len(stimulus_table)
    b = np.zeros(len(stimulus_table), dtype=np.bool)
    for yi in yes_set:
        b[yi] = True

    if debug_plots == True:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,2)
        # ax[0].plot(dff_trace)
        for key, val in debug_dict.iteritems():
            ti, trace = val
            if ti in no_set:
                ax[0].plot(np.arange(key, key+len(trace)), trace, 'b')
            elif ti in yes_set:
                ax[0].plot(np.arange(key, key + len(trace)), trace, 'r', linewidth=2)
            else:
                raise Exception

        for ii in yes_set:
            ax[1].plot([var_dict[ii][2]], [var_dict[ii][3]], 'r.')

        for ii in no_set:
            ax[1].plot([var_dict[ii][2]], [var_dict[ii][3]], 'b.')

        print 'number_of_events:', b.sum()
        plt.show()

    return b

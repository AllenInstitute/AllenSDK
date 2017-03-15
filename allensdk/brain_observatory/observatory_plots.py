import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import scipy.interpolate as si
from scipy.stats.mstats import zscore

import allensdk.brain_observatory.circle_plots as cplots
from contextlib import contextmanager

import numpy as np

SI_RANGE = [ 0, 1.5 ]
P_VALUE_MAX = 0.05
PEAK_DFF_MIN = 3
N_HIST_BINS = 50
STIM_COLOR = "#ccccdd"
STIMULUS_COLOR_MAP = LinearSegmentedColormap.from_list('default',[ [1.0,1.0,1.0,0.0], [.6,.6,.85,1.0] ])
EVOKED_COLOR = "#b30000"
SPONTANEOUS_COLOR = "#0000b3"
LSN_RF_ON_COLOR_MAP = LinearSegmentedColormap.from_list('default', [[0.0,0.0,1.0,1.0],[1,1,1,1],[1.0,0,0.0,1]])
LSN_RF_ON_COLOR_MAP.set_bad((1,1,1,1),1.0)
LSN_RF_OFF_COLOR_MAP = LinearSegmentedColormap.from_list('default', [[0.0,0.0,1.0,1.0],[1,1,1,1],[1.0,0,0.0,1]])
LSN_RF_OFF_COLOR_MAP.set_bad((1,1,1,1),1.0)

def plot_condition_histogram(vals, bins, color=STIM_COLOR):
    plt.grid()
    n, hbins, patches = plt.hist(vals, 
                                 bins=np.arange(len(bins)+1)+1, 
                                 align='left', 
                                 normed=False, 
                                 rwidth=.8, 
                                 color=color,
                                 zorder=3)
    plt.xticks(hbins[:-1], np.round(bins, 2))
   

def plot_selectivity_cumulative_histogram(sis, 
                                          xlabel,
                                          si_range=SI_RANGE, 
                                          n_hist_bins=N_HIST_BINS, 
                                          color=STIM_COLOR):
    bins = np.linspace(si_range[0], si_range[1], n_hist_bins)
    yticks = np.linspace(0,1,5)
    xticks = np.linspace(si_range[0], si_range[1], 4)
         
    yscale = 1.0
    # this is for normalizing to total # cells, not just significant cells
    # yscale = float(num_cells) / len(osis)

    # orientation selectivity cumulative histogram
    n, bins, patches = plt.hist(sis, normed=True, bins=bins, 
                                cumulative=True, histtype='stepfilled', 
                                color=color)
    plt.xlim(si_range)
    plt.ylim([0,yscale])
    plt.yticks(yticks*yscale, yticks)
    plt.xticks(xticks)
    
    plt.xlabel(xlabel)
    plt.ylabel("fraction of cells")
    plt.grid()

def plot_radial_histogram(angles,
                          counts,
                          all_angles=None,
                          include_labels=False,
                          offset=180.0,
                          direction=-1,
                          closed=False,
                          color=STIM_COLOR):
    if all_angles is None:
        all_angles = angles

    dth = (all_angles[1] - all_angles[0]) * 0.5

    max_count = max(counts)

    wedges = []
    for count, angle in zip(counts, angles):
        angle = angle*direction + offset
        wedge = mpatches.Wedge((0,0), count, angle-dth, angle+dth)
        wedges.append(wedge)

    wedge_coll = PatchCollection(wedges)
    wedge_coll.set_facecolor(color)
    wedge_coll.set_zorder(2)

    angles_rad = (all_angles*direction + offset)*np.pi/180.0

    if closed:
        border_coll = cplots.radial_circles([max_count])
    else:
        border_coll = cplots.radial_arcs([max_count], 
                                         min(angles_rad), 
                                         max(angles_rad))
    border_coll.set_facecolor((0,0,0,0))
    border_coll.set_zorder(1)

    line_coll = cplots.angle_lines(angles_rad, 0, max_count)
    line_coll.set_edgecolor((0,0,0,1))
    line_coll.set_linestyle(":")
    line_coll.set_zorder(1)

    ax = plt.gca()
    ax.add_collection(wedge_coll)
    ax.add_collection(border_coll)
    ax.add_collection(line_coll)

    if include_labels:
        cplots.add_angle_labels(ax, angles_rad, all_angles.astype(int), max_count, (0,0,0,1), offset=max_count*0.1)
        ax.set(xlim=(-max_count*1.2, max_count*1.2),
               ylim=(-max_count*1.2, max_count*1.2),
               aspect=1.0)
    else:
        ax.set(xlim=(-max_count*1.05, max_count*1.05),
               ylim=(-max_count*1.05, max_count*1.05),
               aspect=1.0)
        
def plot_time_to_peak(msrs, ttps, t_start, t_end, stim_start, stim_end, cmap):
    plt.plot(ttps, np.arange(msrs.shape[0],0,-1)-0.5, color='black')
    plt.imshow(msrs, 
               cmap=cmap, clim=[0,3], 
               aspect=float((t_end-t_start) / msrs.shape[0]),  # float to get rid of MPL error
               extent=[t_start, t_end, 0, msrs.shape[0]], interpolation='nearest')
    plt.xlim([t_start, t_end])
    plt.ylim([0,msrs.shape[0]])

    plt.axvline(stim_start, linestyle=':', color='black')
    plt.axvline(stim_end, linestyle=':', color='black')

    xticks = np.array([ t_start, stim_start, stim_end, t_end ]) 
    plt.xticks(xticks, np.round(xticks - stim_start, 2))
    plt.xlabel("time from stimulus start (s)")

    yticks, _ = plt.yticks()
    plt.ylabel("cell number")

@contextmanager
def figure_in_px(w, h, file_name, dpi=96.0):
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)

    yield fig
  
    plt.savefig(file_name, dpi=dpi)
    plt.close()

def finalize_no_axes():
    plt.axis('off')
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

def finalize_with_axes():
    plt.tight_layout(pad=.3)

def finalize_no_labels():
    ax = plt.gca()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout(pad=.3)


def plot_combined_speed(binned_resp_vis, binned_dx_vis, binned_resp_sp, binned_dx_sp,
                        evoked_color, spont_color):
    ax = plt.gca()
    num_bins = max(binned_dx_vis.shape[0], binned_dx_sp.shape[0])
    
    plot_speed(binned_resp_vis, binned_dx_vis, num_bins, evoked_color)
    plot_speed(binned_resp_sp, binned_dx_sp, num_bins, spont_color)
    
    xmin = min(binned_dx_vis[:,0].min(), binned_dx_sp[:,0].min())
    xmax = max(binned_dx_vis[:,0].max(), binned_dx_sp[:,0].max())


    ymin = min(binned_resp_vis[:,0].min(), binned_resp_sp[:,0].min())
    ymax = max(binned_resp_vis[:,0].max(), binned_resp_sp[:,0].max())

    xpadding = (xmax-xmin)*.05
    ypadding = (ymax-ymin)*.20

    ax.set_xlim([xmin - xpadding, xmax + xpadding])
    ax.set_ylim([ymin - ypadding, ymax + ypadding])


def plot_speed(binned_resp, binned_dx, num_bins, color): 
    ax = plt.gca()

    # plot the zero bin as a dot with whiskers
    ax.errorbar([ binned_dx[0,0] ], [ binned_resp[0,0] ], yerr=[ binned_resp[0,1] ], fmt='o', color=color)

    # if there's only one bin, drop out
    if len(binned_dx[:,0]) <= 1:
        return

    f = si.interp1d(binned_dx[:,0], binned_resp[:,0])    
    x = np.linspace(min(binned_dx[:,0]), max(binned_dx[:,0]), num=num_bins, endpoint=True)
    y = f(x)
    
    f_up = si.interp1d(binned_dx[:,0], binned_resp[:,0] + binned_resp[:,1])
    y_up = f_up(x)
    
    f_down = si.interp1d(binned_dx[:,0], binned_resp[:,0] - binned_resp[:,1])
    y_down = f_down(x)
    
    ax.plot(x, y, color=color)    
    ax.fill_between(x, y_down, y_up, facecolor=color, alpha=0.1)

def plot_receptive_field(rf, color_map=None, zlim=[-3,3], mask=None):
    if color_map is None:
        color_map = LSN_RF_ON_COLOR_MAP if on else LSN_RF_OFF_COLOR_MAP

    zrf = zscore(rf)

    if mask is not None:
        zrf = np.ma.array(zrf, mask=~mask)

    plt.imshow(zrf, interpolation='nearest', cmap=color_map, clim=zlim)



import matplotlib
matplotlib.use('agg')

import allensdk.core.lims_utilities as lims_utilities
import allensdk.core.json_utilities as json_utilities

from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.ephys.extract_cell_features import get_ramp_stim_characteristics, get_square_stim_characteristics, exp_curve, fit_membrane_tau

import sys
import argparse
import os
import json
import h5py
import numpy as np
from scipy.optimize import curve_fit


import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def load_experiment(file_name, sweep_number):
    ds = NwbDataSet(file_name)
    sweep = ds.get_sweep(sweep_number)
    
    v = sweep['response'] * 1e3
    i = sweep['stimulus'] * 1e12
    dt = 1.0 / sweep['sampling_rate']
    t = np.arange(0, len(v)) * dt

    return (v, i, t)

def plot_single_ap_values(nwb_file, sweeps, features, type_name, figsize):
    figs = [ plt.figure(figsize=(figsize,figsize)) for f in range(3+len(sweeps)) ]

    cw_general = features["specimens"][0]["ephys_features"][0]
    v, i, t = load_experiment(nwb_file, int(sweeps[0]["sweep_num"]))
    if type_name == "short_square" or type_name == "long_square":
        stim_start, stim_dur, stim_amp, start_idx, end_idx = get_square_stim_characteristics(i, t)
    elif type_name == "ramp":
        stim_start, start_idx = get_ramp_stim_characteristics(i, t)

    gen_features = ["threshold", "peak", "trough", "fast_trough", "slow_trough"]
    voltage_features = ["threshold_v", "f_peak", "trough_v", "f_fast_ahp_v", "f_slow_ahp"]
    time_features = ["threshold_t", "f_peak_t", "trough_t", "f_fast_ahp_t", "f_slow_ahp_t"]
    long_square_voltage_features = ["thresh_v", "peak_v", "trough_v", "fast_trough_v", "slow_trough_v"]
    long_square_time_features = ["thresh_t", "peak_t", "trough_t", "fast_trough_t", "slow_trough_t"]

    for s in sweeps:
        spikes = features["specimens"][0]["sweep_ephys_features"][str(s["sweep_num"])]["mean"]["spikes"]
        if (len(spikes) < 1):
            print "NO SPIKES IN SWEEP" + s["sweep_num"]
            continue
        if type_name != "long_square":
            voltages = [spikes[0][f] for f in voltage_features]
            times = [spikes[0][f] - stim_start for f in time_features]
        else:
            voltages = [features["specimens"][0]["cell_ephys_features"]["long_squares"]["rheo_spike_0"][f] for f in long_square_voltage_features]
            times = [features["specimens"][0]["cell_ephys_features"]["long_squares"]["rheo_spike_0"][f] for f in long_square_time_features]

        plt.figure(figs[0].number)
        plt.scatter(range(len(voltages)), voltages, color='gray')

        plt.figure(figs[1].number)
        plt.scatter(range(len(times)), times, color='gray')

        plt.figure(figs[2].number)
        plt.scatter([0], [spikes[0]['upstroke'] / (-spikes[0]['downstroke'])], color='gray')


    plt.figure(figs[0].number)
    
    yvals = [float(cw_general[k + "_v_" + type_name]) for k in gen_features if cw_general[k + "_v_" + type_name] is not None]
    xvals = range(len(yvals))
    
    plt.scatter(xvals, yvals, color='blue', marker='_', s=40, zorder=100)
    plt.xticks(xvals, ['thr', 'pk', 'tr', 'ftr', 'str'])
    plt.title(type_name + ": voltages")

    plt.figure(figs[1].number)
    yvals = [float(cw_general[k + "_t_" + type_name]) for k in gen_features if cw_general[k + "_t_" + type_name] is not None]
    xvals = range(len(yvals))
    plt.scatter(xvals, yvals, color='blue', marker='_', s=40, zorder=100)
    plt.xticks(xvals, ['thr', 'pk', 'tr', 'ftr', 'str'])
    plt.title(type_name + ": times")
    
    plt.figure(figs[2].number)
    if cw_general["upstroke_downstroke_ratio_" + type_name] is not None:
        plt.scatter([0], [float(cw_general["upstroke_downstroke_ratio_" + type_name])], color='blue', marker='_', s=40, zorder=100)
    plt.xticks([])
    plt.title(type_name + ": up/down")

    for index, s in enumerate(sweeps):
        plt.figure(figs[3 + index].number)

        v, i, t = load_experiment(nwb_file, int(s["sweep_num"]))
        plt.plot(t, v, color='black')
        plt.title(s["sweep_num"])

        spikes = features["specimens"][0]["sweep_ephys_features"][str(s["sweep_num"])]["mean"]["spikes"]
        if type_name != "long_square":
            if (len(spikes) < 1):
                print "NO SPIKES IN SWEEP" + s["sweep_num"]
                continue
            voltages = [spikes[0][f] for f in voltage_features]
            times = [spikes[0][f] for f in time_features]
        else:
            voltages = [features["specimens"][0]["cell_ephys_features"]["long_squares"]["rheo_spike_0"][f] for f in long_square_voltage_features]
            times = [features["specimens"][0]["cell_ephys_features"]["long_squares"]["rheo_spike_0"][f] + stim_start for f in long_square_time_features]
        plt.scatter(times, voltages, color='red', zorder=20)
        
        delta_v = 5.0
        plt.plot([spikes[0]['upstroke_t'] - 1e-3 * (delta_v / spikes[0]['upstroke']),
                  spikes[0]['upstroke_t'] + 1e-3 * (delta_v / spikes[0]['upstroke'])], 
                 [spikes[0]['upstroke_v'] - delta_v, spikes[0]['upstroke_v'] + delta_v], color='red')

        plt.plot([spikes[0]['downstroke_t'] - 1e-3 * (delta_v / spikes[0]['downstroke']),
                  spikes[0]['downstroke_t'] + 1e-3 * (delta_v / spikes[0]['downstroke'])], 
                 [spikes[0]['downstroke_v'] - delta_v, spikes[0]['downstroke_v'] + delta_v], color='red')

        if type_name == "ramp":
            plt.xlim(spikes[0]["threshold_t"] - 0.002, spikes[0]["f_slow_ahp_t"] + 0.002)
        elif type_name == "short_square":
            plt.xlim(stim_start - 0.002, stim_start + stim_dur + 0.01)
        elif type_name == "long_square":
            plt.xlim(times[0]- 0.002, times[-1] + 0.002)

    return figs


def plot_qc_figures(nwb_file, features, image_dir, figsize, suffix):
    image_files = {}

    sns.set_style()
    
    cw_detail = features["specimens"][0]["cell_ephys_features"]
    cw_general = features["specimens"][0]["ephys_features"][0]
    
    # Need to figure out maximum width of featureplot before starting
    # Plotting them as individual figures for now    

    print "saving tau and vi figs"
    image_file_set = []

    # 0a - Plot VI curve and linear fit, along with vrest
    x = np.array([d['amp'] for d in cw_detail["long_squares"]["subthresh"]])
    y = np.array([d['peak'] for d in cw_detail["long_squares"]["subthresh"]])
    i = np.array([d['amp'] for d in cw_detail["long_squares"]["subthresh"] if d['amp'] < 0 and d['amp'] > -100])

    fig = plt.figure(figsize=(figsize,figsize))
    plt.scatter(x, y, color='black')
    plt.plot([x.min(), x.max()], [cw_general["vrest"], cw_general["vrest"]], color="blue", linewidth=2)
    plt.plot(i, i * 1e-3 * cw_general["ri"] + cw_general["vrest"], color="red", linewidth=2)
    plt.xlabel("pA")
    plt.ylabel("mV")
    plt.title("ri = {:.1f}, vrest = {:.1f}".format(cw_general["ri"], cw_general["vrest"]))
    image_file = os.path.join(image_dir, "VI_curve%s.png" % suffix)
    image_file_set.append(image_file)
    plt.savefig(image_file, bbox_inches="tight")
    plt.close()
    
    # 0b - Plot tau curve and average
    fig = plt.figure(figsize=(figsize, figsize))
    x = np.array([d['amp'] for d in cw_detail["long_squares"]["subthresh"]])
    y = np.array([d['tau'] for d in cw_detail["long_squares"]["subthresh"]])
    plt.scatter(x, y, color='black')
    i = np.array([d['amp'] for d in cw_detail["long_squares"]["subthresh"] if d['amp'] < 0 and d['amp'] > -100])
    plt.plot([i.min(), i.max()], [cw_detail["long_squares"]["tau"], cw_detail["long_squares"]["tau"]], color="red", linewidth=2)
    plt.xlabel("pA")
    ylim = plt.ylim()
    plt.ylim(0, ylim[1])
    plt.ylabel("tau (ms)")
    image_file = os.path.join(image_dir, "tau_curve%s.png" % suffix)
    image_file_set.append(image_file)
    plt.savefig(image_file, bbox_inches="tight")
    plt.close()
    
    subthresh_dict = {d['sweep_num']: d for d in cw_detail["long_squares"]["subthresh"]}

    # 0c - Plot the subthreshold squares
    tau_sweeps = np.array([d['sweep_num'] for d in cw_detail["long_squares"]["subthresh"] if d['amp'] < 0 and d['amp'] > -100])
    tau_figs = [ plt.figure(figsize=(figsize,figsize)) for i in range(len(tau_sweeps)) ]

    for index, s in enumerate(tau_sweeps):
        v, i, t = load_experiment(nwb_file, s)
        
        plt.figure(tau_figs[index].number)
        
        plt.plot(t, v, color="black")

        if index == 0:
            min_y, max_y = plt.ylim()
        else:
            ylims = plt.ylim()
            if min_y > ylims[0]:
                min_y = ylims[0]
            if max_y < ylims[1]:
                max_y = ylims[1]

        stim_start, stim_dur, stim_amp, start_idx, end_idx = get_square_stim_characteristics(i, t)
        plt.xlim(stim_start - 0.05, stim_start + stim_dur + 0.05)
        plt.scatter([subthresh_dict[s]['peak_t']], [subthresh_dict[s]['peak']], color='red', zorder=10)
        peak_idx = subthresh_dict[s]['peak_idx']
        tenpct_idx, popt = fit_membrane_tau(v, t, start_idx, peak_idx)
        plt.title(str(s))
        if tenpct_idx is np.nan:
            print "Failed to fit tau for sweep ", s
            continue
        if abs((1 / popt[1]) * 1e3 - subthresh_dict[s]['tau']) > 1e-6:
            print "New fit tau of {:g} differs from value in JSON of {:g} for sweep {:d}".format(1 / popt[1] * 1e3, subthresh_dict[s]['tau'], s)
        plt.plot(t[tenpct_idx:peak_idx], exp_curve(t[tenpct_idx:peak_idx] - t[tenpct_idx], *popt), color='blue')

    for index, s in enumerate(tau_sweeps):
        plt.figure(tau_figs[index].number)
        plt.ylim(min_y, max_y)

    for index, tau_fig in enumerate(tau_figs):
        plt.figure(tau_figs[index].number)
        image_file = os.path.join(image_dir, "tau_%d%s.png" % (index, suffix))
        image_file_set.append(image_file)
        tau_fig.savefig(image_file, bbox_inches="tight")
        plt.close()

    image_files['subthreshold_long_squares'] = image_file_set


    # 1 - Plot the short_squares
    print "saving short square figs"
    repeat_amp = cw_detail["short_squares"]["repeat_amp"]
    sweep_features = features["specimens"][0]["sweep_ephys_features"]
    short_squares_sweeps = [s for s in cw_detail["short_squares"]["sweep_info"] 
                            if s["stim_amp"] == repeat_amp and (len(features["specimens"][0]["sweep_ephys_features"][str(s["sweep_num"])]["mean"]["spikes"]) > 0)]
    figs = plot_single_ap_values(nwb_file, short_squares_sweeps, features, "short_square", figsize) 
    image_file_set = []
    for index, fig in enumerate(figs):
        plt.figure(fig.number)
        plt.tight_layout()
        image_file = os.path.join(image_dir, "short_squares_%d%s.png" % (index, suffix))
        image_file_set.append(image_file)
        plt.savefig(image_file, bbox_inches="tight")
        plt.close()    

    image_files['short_squares'] = image_file_set

    # 2 - plot ramps
    print "saving ramps"
    ramps_sweeps = cw_detail["ramps"]["sweep_info"]

    if len(ramps_sweeps) == 0: # no spikes evoked on ramps, but ramps may still be passing -- need to grab them
        print "No passing ramps had spikes"
        ephys_sweeps = features["specimens"][0]["ephys_sweeps"]
        ramps_sweeps = [s["sweep_number"] for s in ephys_sweeps if s["workflow_state"].endswith("passed") and s["ephys_stimulus"]["description"][:10] == "C1RP25PR1S"]
        print "Passing ramps: ", ramps_sweeps

    figs = []
    if len(ramps_sweeps) > 0:
        figs = plot_single_ap_values(nwb_file, ramps_sweeps, features, "ramp", figsize)    

        image_file_set = []
        for index, fig in enumerate(figs):
            plt.figure(fig.number)
            plt.tight_layout()
            image_file = os.path.join(image_dir, "ramps_%d%s.png" % (index, suffix))
            image_file_set.append(image_file)
            plt.savefig(image_file, bbox_inches="tight")
            plt.close()           

        image_files['ramps'] = image_file_set

    # 3 - plot rheo for spike features
    print "saving rheo figs"
    rheo_sweeps = [{"sweep_num": str(cw_general["rheobase_sweep_num"])}]
    figs = plot_single_ap_values(nwb_file, rheo_sweeps, features, "long_square", figsize)

    image_file_set = []
    for index, fig in enumerate(figs):
        plt.figure(fig.number)
        plt.tight_layout()
        image_file = os.path.join(image_dir, "rheo_%d%s.png" % (index, suffix))
        image_file_set.append(image_file)
        plt.savefig(image_file, bbox_inches="tight")
        plt.close()           

    image_files['rheo'] = image_file_set

    # 4 - plot hero sweep and info
    print "saving thumbnail figs"
    image_file_set = []

    plt.figure(figsize=(figsize * 2, figsize))
    v, i, t = load_experiment(nwb_file, int(cw_general["thumbnail_sweep_num"]))
    plt.plot(t, v, color='black')
    stim_start, stim_dur, stim_amp, start_idx, end_idx = get_square_stim_characteristics(i, t)
    plt.xlim(stim_start - 0.05, stim_start + stim_dur + 0.05)
    plt.ylim(-110, 50)
    spike_times = [spk['t'] for spk in features["specimens"][0]["sweep_ephys_features"][str(cw_general["thumbnail_sweep_num"])]["mean"]["spikes"]]
    isis = np.diff(np.array(spike_times))
    plt.title("thumbnail {:d}, amp = {:.1f}".format(cw_general["thumbnail_sweep_num"], stim_amp))
    
    plt.tight_layout()
    image_file = os.path.join(image_dir, "thumbnail_0%s.png" % suffix)
    image_file_set.append(image_file)
    plt.savefig(image_file, bbox_inches="tight")
    plt.close()    

    plt.figure(figsize=(figsize,figsize))
    plt.plot(range(len(isis)), isis)
    plt.ylabel("ISI (ms)")
    if "adaptation" in cw_general:
        plt.title("adapt = {:.3g}".format(cw_general["adaptation"]))
    else:
        plt.title("adapt = not defined")
    
    for k in ["has_delay", "has_burst", "has_pause"]:
        if k not in cw_general:
            cw_general[k] = False

    plt.tight_layout()
    image_file = os.path.join(image_dir, "thumbnail_1%s.png" % suffix)
    image_file_set.append(image_file)
    plt.savefig(image_file, bbox_inches="tight")
    plt.close()    
        
    yvals = [
        float(cw_general["has_delay"]),
        float(cw_general["has_burst"]),
        float(cw_general["has_pause"]),
    ]
    xvals = range(len(yvals))

    plt.figure(figsize=(figsize,figsize))
    plt.scatter(xvals, yvals, color='red')
    plt.xticks(xvals, ['Delay', 'Burst', 'Pause'])
    plt.title("flags")

    plt.tight_layout()
    image_file = os.path.join(image_dir, "thumbnail_2%s.png" % suffix)
    image_file_set.append(image_file)
    plt.savefig(image_file, bbox_inches="tight")
    plt.close()    

    image_files['thumbnail'] = image_file_set

    # 5 - plot fI curve and linear fit
    plt.figure(figsize=(figsize * 2, figsize))
    fI_sorted = sorted(cw_detail["long_squares"]["fI"], key=lambda d: d[0])
    x = [d[0] for d in fI_sorted]
    y = [d[1] for d in fI_sorted]
    last_zero_idx = np.nonzero(y)[0][0] - 1    
    plt.scatter(x, y, color='black')
    plt.plot(x[last_zero_idx:], cw_detail["long_squares"]["fI_fit_slope"] * (np.array(x[last_zero_idx:]) - x[last_zero_idx]), color='red')
    plt.xlabel("pA")
    plt.ylabel("spikes/sec")
    plt.title("slope = {:.3g}".format(cw_general["f_i_curve_slope"]))
    rheo_hero_sweeps = [int(cw_general["rheobase_sweep_num"]), int(cw_general["thumbnail_sweep_num"])]
    rheo_hero_x = []
    for s in rheo_hero_sweeps:
        v, i, t = load_experiment(nwb_file, s)
        stim_start, stim_dur, stim_amp, start_idx, end_idx = get_square_stim_characteristics(i, t)
        rheo_hero_x.append(stim_amp)
    rheo_hero_y = [len(features["specimens"][0]["sweep_ephys_features"][str(s)]["mean"]["spikes"]) for s in rheo_hero_sweeps]
    plt.scatter(rheo_hero_x, rheo_hero_y, zorder=20)
    plt.tight_layout()
    image_file = os.path.join(image_dir, "fi_curve%s.png" % suffix)
    image_files['fi_curve'] = [ image_file ]
    plt.savefig(image_file, bbox_inches="tight")
    plt.close()    

    # 6 - plot sag
    plt.figure(figsize=(figsize * 2, figsize))
    for d in cw_detail["long_squares"]["subthresh"]:
        if d['peak'] == cw_general["vm_for_sag"]:
            v, i, t = load_experiment(nwb_file, int(d['sweep_num']))
            stim_start, stim_dur, stim_amp, start_idx, end_idx = get_square_stim_characteristics(i, t)
            plt.plot(t, v, color='black')
            plt.scatter(d['peak_t'], d['peak'], color='red', zorder=10)
            plt.plot([stim_start + stim_dur - 0.1, stim_start + stim_dur], [d['steady'], d['steady']], color='red', zorder=10)
    plt.xlim(stim_start - 0.25, stim_start + stim_dur + 0.25)
    plt.title("sag = {:.3g}".format(cw_general['sag']))
    plt.tight_layout()
    image_file = os.path.join(image_dir, "sag%s.png" % suffix)
    image_files['sag'] = [ image_file ]
    plt.savefig(image_file, bbox_inches="tight")
    plt.close()    

    return image_files

def make_image_html(small_image_files, large_image_files, index_file):
    html = "<html><body>"
    for image_file_set_name in small_image_files:
        html += "<h3>%s</h3>" % image_file_set_name

        image_files = small_image_files[image_file_set_name]

        for i in xrange(len(image_files)):
            html += "<a href='%s' target='_blank'><img src='%s'></img></a>" % ( os.path.basename(large_image_files[image_file_set_name][i]), 
                                                                os.path.basename(small_image_files[image_file_set_name][i]) )
    html += ("</body></html>")

    with open(index_file, 'w') as f:
        f.write(html)

def make_qc_page(nwb_file, features, working_dir):
    small_image_files = plot_qc_figures(nwb_file, features, working_dir, 2.0, '_small')
    large_image_files = plot_qc_figures(nwb_file, features, working_dir, 5.0, '_large')

    make_image_html(small_image_files, large_image_files, os.path.join(working_dir, 'index.html'))

def main():
    parser = argparse.ArgumentParser(description='analyze specimens for cell-wide features')
    parser.add_argument('nwb_file')
    parser.add_argument('feature_json')
    parser.add_argument('--output_directory', default='.')

    args = parser.parse_args()
    
    features = json_utilities.read(args.feature_json)

    make_qc_page(args.nwb_file, features, args.output_directory)


if __name__ == '__main__': main()

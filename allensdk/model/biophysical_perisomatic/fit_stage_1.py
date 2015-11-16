import os, sys
import ephys_utils
import check_fi_shift
import pandas as pd
import numpy as np
from collections import Counter
import subprocess

from allensdk.ephys.feature_extractor import EphysFeatureExtractor, EphysFeatures
import allensdk.core.json_utilities as ju
from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.model.biophysical_perisomatic.optimize

#from allensdk.core.nwb_data_set import NwbDataSet

SEEDS = [1234, 1001, 4321, 1024, 2048]
FIT_BASE_DIR = os.path.join(os.path.dirname(__file__), "fits")
APICAL_DENDRITE_TYPE = 4
OPTIMIZE_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'optimize.py'))
#MPIEXEC = '/shared/utils.x86_64/hydra/bin/mpiexec'
MPIEXEC = 'mpiexec'

def find_core1_trace(data_set, all_sweeps):
    sweep_type = "C1LSCOARSE"
    _, sweeps, statuses = ephys_utils.get_sweeps_of_type(sweep_type, all_sweeps)
    sweep_status = dict(zip(sweeps, statuses))
    features = EphysFeatureExtractor()

    sweep_info = {}
    for s in sweeps:
        if sweep_status[s][-6:] == "failed":
            continue
        v, i, t = ephys_utils.get_sweep_v_i_t_from_set(data_set, s)
        if np.all(v[-100:] == 0): # Check for early termination of sweep
            continue
        stim_start, stim_dur, stim_amp, start_idx, end_idx = ephys_utils.get_step_stim_characteristics(i, t)
        features.process_instance(s, v, i, t, stim_start, stim_dur, "")
        if "ISICV" in features.feature_list[-1].mean.keys():
            isi_cv = features.feature_list[-1].mean["ISICV"]
        else:
            isi_cv = np.nan
        sweep_info[s] = {"amp": stim_amp, "n_spikes": features.feature_list[-1].mean["n_spikes"], "quality": is_trace_good_quality(v, i, t), "isi_cv": isi_cv }
    
    rheobase_amp = 1e12
    for s in sweep_info:
        if sweep_info[s]["amp"] < rheobase_amp and sweep_info[s]["n_spikes"] > 0:
            rheobase_amp = sweep_info[s]["amp"]
    sweep_to_use_amp = 1e12
    sweep_to_use_isi_cv = 1e121    
    sweep_to_use = -1
    for s in sweep_info:
        if sweep_info[s]["quality"] and sweep_info[s]["amp"] >= 39.0 + rheobase_amp and sweep_info[s]["isi_cv"] < 1.2 * sweep_to_use_isi_cv:
            use_new_sweep = False
            if sweep_to_use_isi_cv > 0.3 and ((sweep_to_use_isi_cv - sweep_info[s]["isi_cv"]) / sweep_to_use_isi_cv) >= 0.2:
                use_new_sweep = True
            elif sweep_info[s]["amp"] < sweep_to_use_amp:
                use_new_sweep = True
            if use_new_sweep:
                print "now using sweep", str(s)
                sweep_to_use = s
                sweep_to_use_amp = sweep_info[s]["amp"]
                sweep_to_use_isi_cv = sweep_info[s]["isi_cv"]
            
    if sweep_to_use == -1:
        print "Could not find appropriate core 1 sweep!"
        return []
    else:
        return [sweep_to_use]

def find_core2_trace(data_set, all_sweeps):
    sweep_type = "C2SQRHELNG"
    _, sweeps, statuses = ephys_utils.get_sweeps_of_type(sweep_type, all_sweeps)
    sweep_status = dict(zip(sweeps, statuses))
    amp_list = []
    core2_amps = {}
    for s in sweeps:
        if sweep_status[s][-6:] == "failed":
            continue
        v, i, t = ephys_utils.get_sweep_v_i_t_from_set(data_set, s)
        if np.all(v[-100:] == 0):
            continue
        stim_start, stim_dur, stim_amp, start_idx, end_idx = ephys_utils.get_step_stim_characteristics(i, t)
        if stim_start is np.nan:
            sweep_status[s] = "manual_failed"
            continue
        core2_amps[s] = stim_amp
        amp_list.append(stim_amp)
    core2_amp_counter = Counter(amp_list)
    common_amps = core2_amp_counter.most_common(3)
    best_amp = 0
    best_n_good = -1
    for amp, count in common_amps:
        n_good = 0
        for k in core2_amps:
            if core2_amps[k] == amp and sweep_status[k][-6:] == "passed":
                v, i, t = ephys_utils.get_sweep_v_i_t_from_set(data_set, k)
                if is_trace_good_quality(v, i, t):
                    n_good += 1
        if n_good > best_n_good:
            best_n_good = n_good
            best_amp = amp
        elif n_good == best_n_good and amp < best_amp:
            best_amp = amp
    if best_n_good <= 1:
        return []
    else:
        for k in core2_amps:
            if core2_amps[k] == best_amp and sweep_status[k][-6:] == "passed":
                sweeps_to_fit.append(k)

def is_trace_good_quality(v, i, t):
    stim_start, stim_dur, stim_amp, start_idx, end_idx = ephys_utils.get_step_stim_characteristics(i, t)
    features = EphysFeatureExtractor()
    features.process_instance("", v, i, t, stim_start, stim_dur, "")

    spikes = features.feature_list[0].mean["spikes"]
    rate = features.feature_list[0].mean["rate"]

    if rate < 5.0:
        return False

    time_to_end = stim_start + stim_dur - spikes[-1]["t"]
    avg_end_isi = ((spikes[-1]["t"] - spikes[-2]["t"]) + (spikes[-2]["t"] - spikes[-3]["t"])) / 2.0

    if time_to_end > 2 * avg_end_isi:
        return False

    isis = np.diff([spk["t"] for spk in spikes])
    if check_for_pause(isis):
        return False

    return True


def check_for_pause(isis):
    if len(isis) <= 2:
        return False
    
    for i, isi in enumerate(isis[1:-1]):
        if isi > 3 * isis[i + 1 - 1] and isi > 3 * isis[i + 1 + 1]:
            return True
    return False


def collect_target_features(ft):
    min_std_dict = {
        'rate': 0.5,
        'adapt': 0.001,
        'f_peak': 2.0,
        'f_trough': 2.0,
        'f_fast_ahp': 2.0,
        'f_slow_ahp': 2.0,
        'f_slow_ahp_time': 0.05,
        'latency': 5.0,
        'ISICV': 0.1,
        'isi_avg': 0.5,
        'doublet': 1.0,
        'time_to_end': 50.0,
        'base_v': 2.0,
        'width': 0.1,
        'upstroke': 5.0,
        'downstroke': 5.0,
        'upstroke_v': 2.0,
        'downstroke_v': 2.0,
        'threshold': 2.0,
        'thresh_ramp': 5.0,
    }

    target_features = []
    for k in ft:
        t = {"name": k, "mean": ft[k]["mean"], "stdev": ft[k]["stdev"]}
        if k in min_std_dict and min_std_dict[k] > ft[k]["stdev"]:
            t["stdev"] = min_std_dict[k]
        target_features.append(t)
    return target_features


def prepare_stage_1(description, passive_fit_data):
    output_directory = description.manifest.get_path('WORKDIR')
    neuronal_model_data = ju.read(description.manifest.get_path('neuronal_model_data'))
    specimen_data = neuronal_model_data['specimen']
    specimen_id = neuronal_model_data['specimen_id']
    is_spiny = not any(t['name'] == u'dendrite type - aspiny' for t in specimen_data['specimen_tags'])
    all_sweeps = specimen_data['ephys_sweeps']
    data_set = NwbDataSet(description.manifest.get_path('stimulus_path'))
    swc_path = description.manifest.get_path('MORPHOLOGY')
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    ra = passive_fit_data['ra']
    cm1 = passive_fit_data['cm1']
    cm2 = passive_fit_data['cm2']

    cap_check_sweeps, _, _ = ephys_utils.get_sweeps_of_type('C1SQCAPCHK', all_sweeps)
    
    # Check for fi curve shift to decide to use core1 or core2
    fi_shift, n_core2 = check_fi_shift.estimate_fi_shift(data_set, cap_check_sweeps)
    fi_shift_threshold = 30.0
    sweeps_to_fit = []
    if abs(fi_shift) > fi_shift_threshold:
        print "FI curve shifted; using Core 1"
        sweeps_to_fit = find_core1_trace(data_set, all_sweeps)
    else:
        sweeps_to_fit = find_core2_trace(data_set, all_sweeps)

        if len(sweeps_to_fit) == 0:
            print "Not enough good Core 2 traces; using Core 1"
            sweeps_to_fit = find_core1_trace(data_set, all_sweeps)

    print "will use sweeps: ", sweeps_to_fit

    jxn = -14.0

    features = EphysFeatureExtractor()
    for s in sweeps_to_fit:
        v, i, t = ephys_utils.get_sweep_v_i_t_from_set(data_set, s)
        v += jxn
        stim_start, stim_dur, stim_amp, start_idx, end_idx = ephys_utils.get_step_stim_characteristics(i, t)
        features.process_instance(s, v, i, t, stim_start, stim_dur, "")
    features.summarize(EphysFeatures("Summary"))
    ft = {}
    for k in features.summary.stdev.keys():
        if k in features.summary.glossary:
            pair = {}
            pair["mean"] = features.summary.mean[k]
            pair["stdev"] = features.summary.stdev[k]
            ft[k] = pair


    # Determine highest step to check for depolarization block
    noise_1_sweeps, _, _ = ephys_utils.get_sweeps_of_type("C1NSSEED_1", all_sweeps)
    noise_2_sweeps, _, _ = ephys_utils.get_sweeps_of_type("C1NSSEED_2", all_sweeps)
    step_sweeps, _, _ = ephys_utils.get_sweeps_of_type("C1LSCOARSE", all_sweeps)
    all_sweeps = noise_1_sweeps + noise_2_sweeps + step_sweeps
    max_i = 0
    for s in all_sweeps:
        try:
            v, i, t = ephys_utils.get_sweep_v_i_t_from_set(data_set, s['sweep_number'])
        except:
            pass
        if np.max(i) > max_i:
            max_i = np.max(i)
    max_i += 10 # add 10 pA
    max_i *= 1e-3 # convert to nA

    # ----------- Generate output and submit jobs ---------------

    # Set up directories
    # Decide which fit(s) we are doing
    if (is_spiny and ft["width"]["mean"] < 0.8) or (not is_spiny and ft["width"]["mean"] > 0.8):
        fit_types = ["f6", "f12"]
    elif is_spiny:
        fit_types = ["f6"]
    else:
        fit_types = ["f12"]

    for fit_type in fit_types:
        fit_type_dir = os.path.join(output_directory, fit_type)
        if not os.path.exists(fit_type_dir):
            os.makedirs(fit_type_dir)
        for seed in SEEDS:
            seed_dir = "{:s}/s{:d}".format(fit_type_dir, seed)
            if not os.path.exists(seed_dir):
                os.makedirs(seed_dir)

    # Collect and save data for target.json file
    target_dict = {}
    target_dict["passive"] = [{
        "ra": ra,
        "cm": { "soma": cm1, "axon": cm1, "dend": cm2 },
        "e_pas": ft['base_v']["mean"]
    }]

    swc_data = pd.read_table(swc_path, sep='\s', comment='#', header=None)
    has_apic = False
    if APICAL_DENDRITE_TYPE in pd.unique(swc_data[1]):
        has_apic = True
        print "Has apical dendrite"
    else:
        print "Does not have apical dendrite"

    if has_apic:
        target_dict["passive"][0]["cm"]["apic"] = cm2

    target_dict["fitting"] = [{
        "junction_potential": jxn,
        "sweeps": sweeps_to_fit,
        "passive_fit_info": passive_fit_data,
        "max_stim_test_na": max_i,        
    }]

    target_dict["stimulus"] = [{
        "amplitude": 1e-3 * stim_amp,
        "delay": 1000.0,
        "duration": 1e3 * stim_dur
    }]

    target_dict["manifest"] = []
    target_dict["manifest"].append({"type": "file", "spec": swc_path, "key": "MORPHOLOGY"})

    target_dict["target_features"] = collect_target_features(ft)

    target_file = os.path.join(output_directory, 'target.json')
    ju.write(target_file, target_dict)

    # Create config.json for each fit type
    config_base_data = ju.read(os.path.join(FIT_BASE_DIR,
                                            'config_base.json'))


    jobs = []
    for fit_type in fit_types:
        config = config_base_data.copy()
        fit_type_dir = os.path.join(output_directory, fit_type)
        config_path = os.path.join(fit_type_dir, "config.json")

        config["biophys"][0]["model_file"] = [ target_file, config_path]
        if has_apic:
            fit_style_file = os.path.join(FIT_BASE_DIR, 'fit_styles', '%s_fit_style.json' + fit_type)
        else:
            fit_style_file = os.path.join(FIT_BASE_DIR, "fit_styles", "%s_noapic_fit_style.json" % fit_type)

        config["biophys"][0]["model_file"].append(fit_style_file)
        config["manifest"].append({"type": "dir", "spec": fit_type_dir, "key": "FITDIR"})
        ju.write(config_path, config)

        for seed in SEEDS:
            logfile = os.path.join(output_directory, fit_type, 's%d' % seed, 'stage_1.log')
            jobs.append({
                    'config_path': os.path.abspath(config_path),
                    'fit_type': fit_type,
                    'log': os.path.abspath(logfile),
                    'seed': seed
                    })
    return jobs


def run_stage_1(jobs):
    for job in jobs:
        args = [MPIEXEC,
                '-np',
                '24',
                sys.executable,
                '-m',
                allensdk.model.biophysical_perisomatic.optimize.__name__,
                str(job['seed']),
                job['config_path']]
        print args
        with open(job['log'], "w") as outfile:
            subprocess.call(args, stdout=outfile)

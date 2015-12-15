import argparse, logging

from scipy import signal
from scipy.optimize import fmin
import numpy as np

import allensdk.core.json_utilities as ju
from allensdk.core.nwb_data_set import NwbDataSet

import find_sweeps as fs
from find_sweeps import MissingSweepException

from find_spikes import find_spikes, find_spikes_list
from spike_cutting import calc_spike_cut_and_v_reset_via_expvar_residuals
from rc import least_squares_simple_circuit_fit_RCEl

from ASGLM import ASGLM_pairwise

from threshold_adaptation import calc_a_b_from_multiblip, err_fix_th
from nonlinearity_parameters import R2R_subthresh_nonlinearity
from MLIN import MLIN

RESTING_POTENTIAL = 'slow_vm_mv'

def load_sweep(file_name, sweep_number, dt=None, cut=0, bessel=False):
    ds = NwbDataSet(file_name)
    data = ds.get_sweep(sweep_number)

    data["dt"] = 1.0 / data["sampling_rate"]

    if cut > 0:
        data["response"] = data["response"][cut:]
        data["stimulus"] = data["stimulus"][cut:]        

    if bessel:
        b, a = signal.bessel(bessel["N"], bessel["Wn"], "low")
        data['response'] = signal.filtfilt(b, a, data['response'], axis=0)

    if dt is not None:
        if data["dt"] != dt:
            data["response"] = subsample_data(data["response"], "mean", data["dt"], dt)
            data["stimulus"] = subsample_data(data["stimulus"], "mean", data["dt"], dt)
            data["start_idx"] = int(data["index_range"][0] / (dt / data["dt"]))
            data["dt"] = dt

    return {
        "voltage": data["response"],
        "current": data["stimulus"],
        "dt": data["dt"],
        "start_idx": data["start_idx"]
        }


def load_sweeps(file_name, sweep_numbers, dt=None, cut=0, bessel=False):
    data = [ load_sweep(file_name, sweep_number, dt, cut, bessel) for sweep_number in sweep_numbers ]

    return {
        'voltage': [ d['voltage'] for d in data ],
        'current': [ d['current'] for d in data ],
        'dt': [ d['dt'] for d in data ],
        'start_idx': [ d['start_idx'] for d in data ],
        } 


def subsample_data(data, method, present_time_step, desired_time_step):
    if present_time_step > desired_time_step:
        raise Exception("you desired times step is smaller than your current time step")

    # number of elements to average over
    n = int(desired_time_step / present_time_step)

    data_subsampled = None

    if method == "mean":
        # if n does not divide evenly into the length of the array, crop off the end
        end = n * int(len(data) / n)
            
        return np.mean(data[:end].reshape(-1,n), 1)

    raise Exception("unknown subsample method: %s" % (method))


def preprocess_neuron(nwb_file, sweep_list, dt=None, cut=0, bessel=None):
    sweep_index = { s['sweep_number']: s for s in sweep_list }

    # load noise
    noise_config = fs.find_noise_sweeps(sweep_index)

    noise1_sweeps = noise_config['noise1']
    logging.debug("noise1_sweeps: %s" % str(noise1_sweeps))
    noise1_data = load_sweeps(nwb_file, noise1_sweeps, dt, cut, bessel)

    if len(noise1_sweeps) == 0:
        raise MissingSweepException("No noise1 sweeps found.")

    if dt is None:
        dt = noise1_data[0]['dt']

    noise2_sweeps = noise_config['noise2']
    logging.debug("noise2_sweeps: %s" % str(noise2_sweeps))
    noise2_data = load_sweeps(nwb_file, noise2_sweeps, dt, cut, bessel)

    # load ramp 
    ramp_config = fs.find_ramp_sweeps(sweep_index)
    ramp_sweeps = ramp_config['suprathreshold']
    logging.debug("ramp_sweeps: %s" % str(ramp_sweeps))
    ramp_data = load_sweeps(nwb_file, ramp_sweeps, dt, cut, bessel)
    
    # load ssqs
    ssq_config = fs.find_short_square_sweeps(sweep_index)

    subthresh_ssq_sweeps = ssq_config['maximum_subthreshold']
    logging.debug("subthresh_ssq_sweeps: %s" % str(subthresh_ssq_sweeps))
    subthresh_ssq_data = load_sweeps(nwb_file, subthresh_ssq_sweeps, dt, cut, bessel)

    superthresh_ssq_sweeps = ssq_config['minimum_suprathreshold']
    logging.debug("suprathresh_ssq_sweeps: %s" % str(superthresh_ssq_sweeps))
    superthresh_ssq_data = load_sweeps(nwb_file, superthresh_ssq_sweeps, dt, cut, bessel)
        
    # load max subthreshold lsq
    lsq_config = fs.find_long_square_sweeps(sweep_index)    
    subthresh_lsq_sweeps = lsq_config['maximum_subthreshold']
    logging.debug("subthresh lsq sweeps: %s" % str(subthresh_lsq_sweeps))
    subthresh_lsq_data = load_sweeps(nwb_file,
                                     [ subthresh_lsq_sweeps[0] ], 
                                     dt, 
                                     cut = 0, bessel = False) # no cut or bessel here
    # load ssq triple
    ssq_triple_sweeps = ssq_config['triple']
    if len(ssq_triple_sweeps):
        logging.debug("ssq triple sweeps: %s" % str(ssq_triple_sweeps))
        ssq_triple_data = load_sweeps(nwb_file, ssq_triple_sweeps, dt, cut, bessel)
        logging.info("Has short square triple")
    else:
        ssq_triple_sweeps = None
        ssq_triple_data = None
        logging.info("No short square triple")

    # load ramp to rheo
    R2R_config = fs.find_ramp_to_rheo_sweeps(sweep_index)
    R2R_sweeps = R2R_config['all']

    if len(R2R_sweeps):
        logging.debug("r2r: %s" % str(R2R_sweeps))
        R2R_data = load_sweeps(nwb_file, R2R_sweeps, dt, cut, bessel)
        logging.info("Has short ramp to rheo")
    else:
        R2R_sweeps = None
        R2R_data = None
        logging.info("No ramp to rheo")



    # compute El
    subthresh_noise_current_list=[]
    subthresh_noise_voltage_list=[]
    noise_El_list=[]
    for ss in range(0, len(noise1_data['current'])):
        # subthreshold noise has first epoch of noise with a region of no stimulation before and after (note the selection of end point is hard coded)
        subthresh_noise_current_list.append(noise1_data['current'][ss][noise1_data['start_idx'][ss]:int(6./dt)])
        subthresh_noise_voltage_list.append(noise1_data['voltage'][ss][noise1_data['start_idx'][ss]:int(6./dt)])
        noise_El_list.append(sweep_index[noise1_sweeps[ss]][RESTING_POTENTIAL]*1e-3)  

    El_noise = np.mean(noise_El_list)
    El_superthreshold_blip = sweep_index[superthresh_ssq_sweeps[0]][RESTING_POTENTIAL]*1e-3

    # compute R & C
    (R_lssq_wrest_list, C_lssq_wrest_list, El_lssq_wrest_list) = \
        least_squares_simple_circuit_fit_RCEl(subthresh_noise_voltage_list, subthresh_noise_current_list, dt)

    R_lssq_wrest_mean = np.mean(R_lssq_wrest_list)
    C_lssq_wrest_mean = np.mean(C_lssq_wrest_list)
    El_lssq_wrest_mean = np.mean(El_lssq_wrest_list)

    # find indices of spikes in noise
    noise1_ind_wo_test_pulse_removed, _ = find_spikes_list(noise1_data['voltage'], dt)
    noise2_ind_wo_test_pulse_removed, _ = find_spikes_list(noise2_data['voltage'], dt)

    # Put all ISI ind in an array
    ISI_length = np.array([])
    for ii in range(len(noise1_ind_wo_test_pulse_removed)):
        ISI_length = np.append(ISI_length, noise1_ind_wo_test_pulse_removed[ii][1:] - noise1_ind_wo_test_pulse_removed[ii][:-1])
    for ii in range(len(noise2_ind_wo_test_pulse_removed)):
        ISI_length = np.append(ISI_length, noise2_ind_wo_test_pulse_removed[ii][1:] - noise2_ind_wo_test_pulse_removed[ii][:-1])
    min_ISI_len = np.min(ISI_length)

    # compute spike cutting
    # putting deltaV as zero here 
    # TODO: corinnet check this!
    (spike_cut_length_NODELTAV, slope_at_min_expVar_list_NODELTAV, intercept_at_min_expVar_list_NODELTAV) \
        = calc_spike_cut_and_v_reset_via_expvar_residuals(noise1_data['current'], noise1_data['voltage'], 
                                                          dt, El_noise, 0, 
                                                          max_spike_cut_time = min_ISI_len * dt,
                                                          MAKE_PLOT=False, 
                                                          SHOW_PLOT=False, 
                                                          BLOCK=False)

    # compute afterspike currents
    k_asc_possible = -np.array([-3, -10., -30., -100., -300.]) 
    (best_k_pair, best_asc_amp, best_R, best_C, best_El, best_llh) = \
        ASGLM_pairwise(k_asc_possible, 
                       noise1_data['current'], noise1_data['voltage'], noise1_ind_wo_test_pulse_removed, 
                       C_lssq_wrest_mean, C_lssq_wrest_mean*R_lssq_wrest_mean, spike_cut_length_NODELTAV, dt, El_noise, 'ascR',
                       SHORT_RUN=False, MAKE_PLOT=False, SHOW_PLOT=False, BLOCK=False)

    asc_amp_from_ASGLM = np.mean(best_asc_amp, axis=0)
    asc_tau_from_ASGLM = 1.0 / best_k_pair

    # compute instantaneous threshold
    try:
        voltage = superthresh_ssq_data['voltage'][0][superthresh_ssq_data['start_idx'][0]:]
        spike_inds, spike_vs = find_spikes(voltage, dt)

        th_inf_via_Vmeasure = spike_vs[0]
        th_inf_via_Vmeasure_from0 = th_inf_via_Vmeasure - El_superthreshold_blip

    except:
        raise MissingSweepException("The superthreshold short square sweep must have a spike, but no spike was detected.")

    # compute adapted threshold
    th_vals_relativeto0 = np.array([])
    th_vals = np.array([])
    for ss in range(0, len(noise1_data['voltage'])):
        if len(noise1_ind_wo_test_pulse_removed[ss])<1:
            th_vals_relativeto0 = np.append(th_vals_relativeto0, np.nan)
            th_vals = np.append(th_vals, np.nan)
        else:
            th_vals_relativeto0 = np.append(th_vals_relativeto0, noise1_data['voltage'][ss][noise1_ind_wo_test_pulse_removed[ss]]-noise_El_list[ss])
            th_vals = np.append(th_vals, noise1_data['voltage'][ss][noise1_ind_wo_test_pulse_removed[ss]])

    th_inf_from_5percentile_noise_from0 = np.percentile(th_vals_relativeto0,5)
    th_adapt_from_95percentile_noise_from0 = np.percentile(th_vals_relativeto0,95)   

    if ssq_triple_data:
        (a_spike_component_of_threshold, b_spike_component_of_threshold, mean_voltage_first_spike_of_blip) = \
            calc_a_b_from_multiblip(ssq_triple_data, dt, 
                                    MAKE_PLOT=False,
                                    SHOW_PLOT=False,
                                    BLOCK=False,
                                    PUBLICATION_PLOT=False)

        thresh_reset_const = a_spike_component_of_threshold        

        #TODO: right now only feeding one voltage trace to this function
        if a_spike_component_of_threshold is None or b_spike_component_of_threshold is None:
            print "threshold component of spike could not be calculated from the multiblip data"
            a_voltage_component_of_threshold = None
            b_voltage_component_of_threshold = None
        else:
            #TODO:this function needs to be changed to use experimental change of reference
            #TODO:confirm this is correct with RAM again
            b_voltage_guess = 5.0         
            a_voltage_guess = 0.1*b_voltage_guess
            #TODO note that currently using th_inf_from_5percentile_noise_from0+El_noise instead of the value calculated using the local noise El
            ab_voltage_component_of_threshold_from_noise = fmin(func=err_fix_th, args=(noise1_data['voltage'][0], El_noise, 
                                                                                       spike_cut_length_NODELTAV, noise1_ind_wo_test_pulse_removed[0], 
                                                                                       th_inf_from_5percentile_noise_from0+El_noise, dt, a_spike_component_of_threshold, 
                                                                                       b_spike_component_of_threshold), x0=[a_voltage_guess,b_voltage_guess])
            a_voltage_component_of_threshold=ab_voltage_component_of_threshold_from_noise[0]
            b_voltage_component_of_threshold=ab_voltage_component_of_threshold_from_noise[1]        
            print "ab_opt_noise", ab_voltage_component_of_threshold_from_noise
    else:
        a_spike_component_of_threshold = None
        b_spike_component_of_threshold = None
        a_voltage_component_of_threshold = None
        b_voltage_component_of_threshold = None 

    if R2R_data:
        R2R_El_list = []
        R2R_voltage_list = []
        R2R_current_list = []
        for ss in range(len(R2R_sweeps)):
            R2R_voltage_list.append(R2R_data['voltage'][ss][R2R_data['start_idx'][ss]:])
            R2R_current_list.append(R2R_data['current'][ss][R2R_data['start_idx'][ss]:])
            R2R_El_list.append(sweep_index[R2R_sweeps[ss]][RESTING_POTENTIAL]*1e-3)  

        (line_param_RV_list, line_param_ElV_list, line_param_RV_all, line_param_ElV_all) = \
            R2R_subthresh_nonlinearity(R2R_current_list, R2R_voltage_list, 
                                       R_lssq_wrest_mean, C_lssq_wrest_mean, El_lssq_wrest_mean, 
                                       R2R_El_list, dt,
                                       MAKE_PLOT=False,
                                       SHOW_PLOT=False, 
                                       BLOCK=False)
    else:
        line_param_RV_list = None
        line_param_ElV_list = None
        line_param_RV_all = None
        line_param_ElV_all = None

    
    subthresh_lsq_voltage = subthresh_lsq_data['voltage'][0][subthresh_lsq_data['start_idx'][0]:]
    subthresh_lsq_current = subthresh_lsq_data['current'][0][subthresh_lsq_data['start_idx'][0]:]
    (var_of_section, sv_for_expsymm, tau_from_AC) = MLIN(subthresh_lsq_voltage, 
                                                         subthresh_lsq_current, 
                                                         R_lssq_wrest_mean, C_lssq_wrest_mean, dt, 
                                                         MAKE_PLOT=False,
                                                         SHOW_PLOT=False, 
                                                         BLOCK=False, 
                                                         PUBLICATION_PLOT=False)

    values = {
        'El_reference': El_noise,
        'El': 0.0,
        'dt': dt,
        'R_input': R_lssq_wrest_mean,
        'C': C_lssq_wrest_mean,
        'spike_cut_length': spike_cut_length_NODELTAV,
        'spike_cutting_intercept': intercept_at_min_expVar_list_NODELTAV,
        'spike_cutting_slope': slope_at_min_expVar_list_NODELTAV,
        'asc_amp_array': asc_amp_from_ASGLM,
        'asc_tau_array': asc_tau_from_ASGLM,
        'th_inf': th_inf_via_Vmeasure_from0,
        'th_adapt': th_adapt_from_95percentile_noise_from0,
        'deltaV': None,
        'threshold_adaptation': {
            'a_spike_component_of_threshold': a_spike_component_of_threshold,
            'b_spike_component_of_threshold': b_spike_component_of_threshold,
            'a_voltage_component_of_threshold': a_voltage_component_of_threshold,
            'b_voltage_component_of_threshold': b_voltage_component_of_threshold,
            },
        'nonlinearity_parameters': {
            'line_param_RV_all':line_param_RV_all,
            'line_param_ElV_all':line_param_ElV_all            
            },
        'spike_inds': { 
            'noise1': noise1_ind_wo_test_pulse_removed,
            'noise2': noise2_ind_wo_test_pulse_removed
            },
        'MLIN': {
            'var_of_section': var_of_section,
            'sv_for_expsymm': sv_for_expsymm,
            'tau_from_AC': tau_from_AC
            },
        'nwb_file': nwb_file
        }

    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nwb_file")
    parser.add_argument("sweep_list_file")
    parser.add_argument("output_json")
    parser.add_argument("--dt", default=DEFAULT_DT)
    parser.add_argument("--bessel", default=DEFAULT_BESSEL)
    parser.add_argument("--cut", default=DEFAULT_CUT)

    args = parser.parse_args()

    sweep_list = ju.read(args.sweep_list_file)
    
    values = preprocess_neuron(args.nwb_file, sweep_list, data_config)

    ju.write(args.output_json, values)
    

if __name__ == "__main__": main()

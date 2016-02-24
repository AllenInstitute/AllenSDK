import os, logging, re
import numpy as np
from preprocess_neuron import preprocess_neuron

import allensdk.core.json_utilities as ju

DATA_CONFIG_FILES = [
    "/data/mat/Corinne/GLIF_subset/data_config_files/329552531_data_config.json",
    "/data/mat/Corinne/GLIF_subset/data_config_files/476106176_data_config.json",
    "/data/mat/Corinne/GLIF_subset/data_config_files/485455285_data_config.json",
    "/data/mat/Corinne/GLIF_subset/data_config_files/369697038_data_config.json",
    "/data/mat/Corinne/GLIF_subset/data_config_files/485912047_data_config.json"
]

TEST_DIR = "/local1/stash/glif_clean/test/dictionaries/"

OUT_DIR = "/local1/stash/allensdk/allensdk/model/glif/test/preprocess"

def assert_equal(v1, v2, message, errs):
    if isinstance(v1, list) or isinstance(v1, np.ndarray):
        if not (v1 == v2).all():
            errs.append("%s: %s != %s" % (message, str(v1), str(v2)))        
    elif v1 != v2:
        if v1 is None or v2 is None:
            errs.append("%s: %s != %s" % (message, str(v1), str(v2)))
        else:
            diff = abs(v1 - v2)
            logging.info("%s: differ by %f pct" % (message, diff / v1 * 100.0))
            
            errs.append("%s: %s != %s" % (message, str(v1), str(v2)))

def test_preprocess_neuron():
    logging.getLogger().setLevel(logging.DEBUG)
    p = re.compile("(\d+)_data_config.json")
    dt = 5e-05
    bessel = { 'N': 4, 'Wn': .1 }
    cut = 0

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


    for data_config_file in DATA_CONFIG_FILES:
        logging.info("testing %s" % data_config_file)
        fname = os.path.basename(data_config_file)
        m = p.match(fname)

        sid = m.groups()

        test_data_file = os.path.join(TEST_DIR, "%s_preprocessed_dict.json" % sid)

        if not os.path.exists(test_data_file):
            logging.warning("no test file %s" % test_data_file)
            continue

        out_file = os.path.join(OUT_DIR, "%s_preprocessed_dict.json" % sid)

        data_config = ju.read(data_config_file)
        nwb_file = data_config["filename"]
        sweep_list = data_config["sweeps"].values()

        d = preprocess_neuron(nwb_file, sweep_list, dt, cut, bessel)

        dictionary = ju.read(test_data_file)

        ju.write(out_file, d)

        errs = []
        assert_equal(d['El'], 0.0, 'El', errs)
        assert_equal(d['El_reference'], dictionary['El']['El_noise']['measured']['mean'], 'El_reference', errs)
        assert_equal(d['deltaV'], None, 'deltaV', errs)
        assert_equal(d['dt'], dictionary['dt_used_for_preprocessor_calculations'], 'dt', errs)
        assert_equal(d['R_input'], dictionary['resistance']['R_lssq_Wrest']['mean'], 'R_input', errs)
        assert_equal(d['C'], dictionary['capacitance']['C_lssq_Wrest']['mean'], 'C', errs)
        assert_equal(d['th_inf'], dictionary['th_inf']['via_Vmeasure']['from_zero'], 'th_inf', errs)
        assert_equal(d['th_adapt'], dictionary['th_adapt']['from_95percentile_noise']['deltaV'], 'th_adapt', errs)
        assert_equal(d['spike_cut_length'], dictionary['spike_cutting']['NOdeltaV']['cut_length'], 'spike_cut_length', errs)
        assert_equal(d['spike_cutting_intercept'], dictionary['spike_cutting']['NOdeltaV']['intercept'], 'spike_cutting_intercept', errs)
        assert_equal(d['spike_cutting_slope'], dictionary['spike_cutting']['NOdeltaV']['slope'], 'spike_cutting_slope', errs)
        assert_equal(d['asc_amp_array'], dictionary['asc']['amp'], 'asc_amp_array', errs)
        assert_equal(d['asc_tau_array'], 1./np.array(dictionary['asc']['k']), 'asc_tau_array', errs)

        nlp = d['nonlinearity_parameters']
        assert_equal(nlp['line_param_RV_all'], dictionary['nonlinearity_parameters']['line_param_RV_all'], 'line_param_RV_all', errs)
        assert_equal(nlp['line_param_ElV_all'], dictionary['nonlinearity_parameters']['line_param_ElV_all'], 'line_param_ElV_all', errs)

        ta = d['threshold_adaptation']
        assert_equal(ta['a_spike_component_of_threshold'], dictionary['threshold_adaptation']['a_spike_component_of_threshold'], 'a_spike', errs)
        assert_equal(ta['b_spike_component_of_threshold'], dictionary['threshold_adaptation']['b_spike_component_of_threshold'], 'b_spike', errs) 
        assert_equal(ta['a_voltage_component_of_threshold'], dictionary['threshold_adaptation']['a_voltage_component_of_threshold'], 'a_voltage', errs)
        assert_equal(ta['b_voltage_component_of_threshold'], dictionary['threshold_adaptation']['b_voltage_component_of_threshold'], 'b_voltage', errs) 

        mlin = d['MLIN']
        assert_equal(mlin['var_of_section'], dictionary['MLIN']['var_of_section'], 'var_of_section', errs)
        assert_equal(mlin['sv_for_expsymm'],  dictionary['MLIN']['sv_for_expsymm'], 'sv_for_expsymm', errs)
        assert_equal(mlin['tau_from_AC'], dictionary['MLIN']['tau_from_AC'], 'tau_from_AC', errs)

        if len(errs) > 0:
            for err in errs:
                logging.error(err)
            raise Exception("Preprocessor outputs did not match.")
            
    

if __name__ == "__main__": test_preprocess_neuron()

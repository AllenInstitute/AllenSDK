import os, sys
#import argparse
import subprocess

import numpy as np

import ephys_utils
import passive_fitting.preprocess as passive_prep

import allensdk.core.json_utilities as ju
from allensdk.model.biophys_sim.config import Config
from allensdk.core.nwb_data_set import NwbDataSet
#from allensdk.core.nwb_data_set import NwbDataSet

PASSIVE_FITTING_DIR = os.path.join(os.path.dirname(__file__), "passive_fitting")

def run_passive_fit(description):
    import sys
    sys.path.append(r'/local1/eclipse/plugins/org.python.pydev_4.3.0.201508182223/pysrc')
    #import pydevd; pydevd.settrace(stdoutToServer=True, stderrToServer=True)

    #specimen_id, data_set, is_spiny, all_sweeps, swc_path, output_directory):
    output_directory = description.manifest.get_path('WORKDIR')
    specimen_data = ju.read(description.manifest.get_path('specimen_data'))
    is_spiny = specimen_data['dendrite type'] != 'aspiny'
    all_sweeps = specimen_data['sweeps'] # TODO: this should probably just be in the lims input file
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cap_check_sweeps, _, _ = \
        ephys_utils.get_sweeps_of_type('C1SQCAPCHK',
                                       all_sweeps)
    
    passive_fit_data = {}

    if len(cap_check_sweeps) > 0:
        data_set = NwbDataSet(description.manifest.get_path('stimulus_path'))
        d = passive_prep.get_passive_fit_data(cap_check_sweeps, data_set);

        grand_up_file = os.path.join(output_directory, 'upbase.dat')
        np.savetxt(grand_up_file, d['grand_up'])

        grand_down_file = os.path.join(output_directory, 'downbase.dat')
        np.savetxt(grand_down_file, d['grand_down'])
        
        passive_fit_data["bridge"] = d['bridge_avg']
        passive_fit_data["escape_time"] = d['escape_t']

        #fit_1_script = os.path.join(PASSIVE_FITTING_DIR, "neuron_passive_fit.py")
        fit_1_file = description.manifest.get_path('fit_1_file')
        try:
            fit_1_data = subprocess.check_output([ sys.executable,
                                                   '-m', 'allensdk.model.deap_optimize.passive_fitting.neuron_passive_fit', 
                                                   str(d['escape_t']),
                                                   os.path.realpath(description.manifest.get_path('manifest')) ])
    #                                               '--specimen_id', str(specimen_id), 
    #                                               '--up_file', grand_up_file, 
    #                                               '--down_file', grand_down_file, 
    #                                               '--swc_path', swc_path, 
    #                                               '--limit', str(d['escape_t']),
    #                                               '--output_file', fit_1_file ])
            passive_fit_data['fit_1'] = ju.read(fit_1_file)
        except:
            print('oh well.')

        fit_2_file = description.manifest.get_path('fit_2_file')
        fit_2_data = subprocess.check_output([ sys.executable,
                                              '-m', 'allensdk.model.deap_optimize.passive_fitting.neuron_passive_fit2',
                                               str(d['escape_t']),
                                               os.path.realpath(description.manifest.get_path('manifest')) ])

#                                               fit_2_script, 
#                                                '--specimen_id', str(specimen_id), 
#                                                '--up_file', grand_up_file, 
#                                                '--down_file', grand_down_file, 
#                                                '--swc_path', swc_path, 
#                                                '--limit', str(d['escape_t']),
#                                                '--output_file', fit_2_file ],
#                                              env={'PYTHONPATH': '/home/timf/git/allensdk:/local1/anaconda/bin:/home/timf/.local/lib/python2.7/site-packages:/home/timf/nrn/nrn/lib/python'})
        passive_fit_data['fit_2'] = ju.read(fit_2_file)

        #fit_3_script = os.path.join(PASSIVE_FITTING_DIR, "neuron_passive_fit_elec.py")
        fit_3_file = os.path.join(output_directory, 'fit_3_data.json')
        fit_3_data = subprocess.check_output([ sys.executable,
                                              '-m', 'allensdk.model.deap_optimize.passive_fitting.neuron_passive_fit_elec',
                                               str(d['escape_t']),
                                               str(d['bridge_avg']),
                                               str(1.0),
                                               os.path.realpath(description.manifest.get_path('manifest')) ])
#                                                '--specimen_id', str(specimen_id), 
#                                                '--up_file', grand_up_file, 
#                                                '--down_file', grand_down_file, 
#                                                '--swc_path', swc_path, 
#                                                '--limit', str(d['escape_t']),
#                                                '--bridge', str(d['bridge_avg']),
#                                                '--elec_cap', str(1.0),
#                                                '--output_file', fit_3_file ])
        passive_fit_data['fit_3'] = ju.read(fit_3_file)
        
        # Check for potentially problematic outcomes
        cm_rel_delta = (passive_fit_data["fit_1"]["Cm"] - passive_fit_data["fit_3"]["Cm"]) / passive_fit_data["fit_1"]["Cm"]
        if passive_fit_data["fit_2"]["err"] < passive_fit_data["fit_1"]["err"]:
            print "Fixed Ri gave better results than original"
            if passive_fit_data["fit_2"]["err"] < passive_fit_data["fit_3"]["err"]:
                print "Using fixed Ri results"
                passive_fit_data["fit_for_next_step"] = passive_fit_data["fit_2"]
            else:
                print "Using electrode results"
                passive_fit_data["fit_for_next_step"] = passive_fit_data["fit_3"]
        elif abs(cm_rel_delta) > 0.1:
            print "Original and electrode fits not in sync:"
            print "original Cm: ", passive_fit_data["fit_1"]["Cm"]
            print "w/ electrode Cm: ", passive_fit_data["fit_3"]["Cm"]
            if passive_fit_data["fit_1"]["err"] < passive_fit_data["fit_3"]["err"]:
                print "Original has lower error"
                passive_fit_data["fit_for_next_step"] = passive_fit_data["fit_1"]
            else:
                print "Electrode has lower error"
                passive_fit_data["fit_for_next_step"] = passive_fit_data["fit_3"]
        else:
            passive_fit_data["fit_for_next_step"] = passive_fit_data["fit_1"]

        ra = passive_fit_data["fit_for_next_step"]["Ri"]
        if is_spiny:
            combo_cm = passive_fit_data["fit_for_next_step"]["Cm"]
            a1 = passive_fit_data["fit_for_next_step"]["A1"]
            a2 = passive_fit_data["fit_for_next_step"]["A2"]
            cm1 = 1.0
            cm2 = (combo_cm * (a1 + a2) - a1) / a2
        else:
            cm1 = passive_fit_data["fit_for_next_step"]["Cm"]
            cm2 = passive_fit_data["fit_for_next_step"]["Cm"]
    else:
        print "No cap check trace found"
        ra = 100.0
        cm1 = 1.0
        if is_spiny:
            cm2 = 2.0
        else:
            cm2 = 1.0

    passive_fit_data['ra'] = ra
    passive_fit_data['cm1'] = cm1
    passive_fit_data['cm2'] = cm2
    
    return passive_fit_data

def main(limit, manifest_path):
    app_config = Config()
    description = app_config.load(manifest_path)

#     parser = argparse.ArgumentParser(description='Set up DEAP-style fit')
#     parser.add_argument('--output_dir', required=True)
#     parser.add_argument('specimen_id', type=int)
#     args = parser.parse_args()
# 
#     data = lims_utils.get_specimen_info(args.specimen_id)
#     output_directory = os.path.join(args.output_dir, 'specimen_%d' % args.specimen_id)
# 
#     is_spiny = data['dendrite type'] != 'aspiny'
# 
#     data_set = NwbDataSet(data['nwb_path'])
# 
#passive_fit_data = run_passive_fit(data['id'], data_set, is_spiny, data['sweeps'], data['swc_path'], output_directory)
    run_passive_fit(description)
#     json_utilities.write(os.path.join(output_directory, 'passive_fit_data.json'), passive_fit_data)
# 
if __name__ == "__main__":
    import sys
    sys.path.append(r'/local1/eclipse/plugins/org.python.pydev_4.3.0.201508182223/pysrc')
    import pydevd; pydevd.settrace(stdoutToServer=True, stderrToServer=True)
    
    limit = sys.argv[-2]
    manifest_path = sys.argv[-1]
    
    main(limit, manifest_path)
    

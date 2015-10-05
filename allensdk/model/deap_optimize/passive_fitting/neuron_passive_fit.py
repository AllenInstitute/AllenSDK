#!/usr/bin/env python

#import allensdk.model.deap_optimize.lims_utils as lims_utils
from neuron import h
import numpy as np
import argparse
#import matplotlib.pyplot as plt
import os
import neuron_utils
from neuron_utils import read_neuron_fit_stdout
#import json
import allensdk.core.json_utilities as json_utilities
from allensdk.model.biophys_sim.config import Config
#from allensdk.core.nwb_data_set import NwbDataSet

#import allensdk.model.deap_optimize.passive_fitting.preprocess as passive_prep
#import allensdk.model.deap_optimize.ephys_utils as ephys_utils

# Load the morphology

BASEDIR = os.path.dirname(__file__)#"/data/mat/nathang/deap_optimize/passive_fitting"


@read_neuron_fit_stdout
def neuron_passive_fit(up_data, down_data, swc_path, limit):
    h.load_file("stdgui.hoc")
    h.load_file("import3d.hoc")
    neuron_utils.load_morphology(swc_path)

    for sec in h.allsec():
        sec.insert('pas')
        for seg in sec:
            seg.pas.e = 0

    h.load_file(os.path.join(BASEDIR, "passive", "fixnseg.hoc"))
    h.load_file(os.path.join(BASEDIR, "passive", "iclamp.ses"))
    h.load_file(os.path.join(BASEDIR, "passive", "params.hoc"))
    h.load_file(os.path.join(BASEDIR, "passive", "mrf.ses"))

    h.v_init = 0
    h.tstop = 100
    h.dt = 0.005

    fit_start = 4.0025

    v_rec = h.Vector()
    t_rec = h.Vector()
    v_rec.record(h.soma[0](0.5)._ref_v)
    t_rec.record(h._ref_t)

    mrf = h.MulRunFitter[0]
    gen0 = mrf.p.pf.generatorlist.object(0)
    gen0.toggle()
    fit0 = gen0.gen.fitnesslist.object(0)

    up_t = h.Vector(up_data[:, 0])
    up_v = h.Vector(up_data[:, 1])
    fit0.set_data(up_t, up_v)
    fit0.boundary.x[0] = fit_start
    fit0.boundary.x[1] = limit
    fit0.set_w()

    gen1 = mrf.p.pf.generatorlist.object(1)
    gen1.toggle()
    fit1 = gen1.gen.fitnesslist.object(0)

    down_t = h.Vector(down_data[:, 0])
    down_v = h.Vector(down_data[:, 1])
    fit1.set_data(down_t, down_v)
    fit1.boundary.x[0] = fit_start
    fit1.boundary.x[1] = limit
    fit1.set_w()

    minerr = 1e12
    for _ in range(3):
        # Need to re-initialize the internal MRF variables, not top-level proxies
        # for randomize() to work
        mrf.p.pf.parmlist.object(0).val = 100
        mrf.p.pf.parmlist.object(1).val = 1
        mrf.p.pf.parmlist.object(2).val = 10000
        mrf.p.pf.putall()
        mrf.randomize()
        mrf.prun()
        if mrf.opt.minerr < minerr:
            fit_Ri = h.Ri
            fit_Cm = h.Cm
            fit_Rm = h.Rm
            minerr = mrf.opt.minerr

    h.region_areas()

    return {
        'Ri': fit_Ri,
        'Cm': fit_Cm,
        'Rm': fit_Rm,
        'err': minerr
        }

def arg_parser():
    parser = argparse.ArgumentParser(description='analyze cap check sweep')
    parser.add_argument('--up_file')
    parser.add_argument('--down_file')
    parser.add_argument('--swc_path')
    parser.add_argument('--specimen_id', type=int, required=True)
    parser.add_argument('--limit', type=float, required=True)
    parser.add_argument('--output_file', required=True)
    return parser


def process_inputs(parser):
    args = parser.parse_args()

#     specimen_info = None

#     if not args.swc_path:
#         specimen_info = lims_utils.get_specimen_info(args.specimen_id)
#         swc_path = specimen_info['swc_path']
#     else:
    swc_path = args.swc_path

#     if not args.up_file or not args.down_file:
#         if not specimen_info:
#             specimen_info = lims_utils.get_specimen_info(args.specimen_id)
#         cap_check_sweeps, _, _ = ephys_utils.get_sweeps_of_type('C1SQCAPCHK', specimen_info['sweeps'])
# 
#         d = passive_prep.get_passive_fit_data(cap_check_sweeps, NwbDataSet(specimen_info['nwb_path']))
#         up_data, down_data = d['grand_up'], d['grand_down']
#     else:
    up_data = np.loadtxt(args.up_file)
    down_data = np.loadtxt(args.down_file)

    return args, up_data, down_data, swc_path

def main():
    import sys
    sys.path.append(r'/local1/eclipse/plugins/org.python.pydev_4.3.0.201508182223/pysrc')
    #import pydevd; pydevd.settrace(stdoutToServer=True, stderrToServer=True)
    
    manifest_path = sys.argv[-1]
    limit = float(sys.argv[-2])
    os.chdir(os.path.dirname(manifest_path))
    app_config = Config()
    description = app_config.load(manifest_path)
    
#    parser = arg_parser()

    upfile = description.manifest.get_path('upfile')
    up_data =  np.loadtxt(upfile)
    downfile = description.manifest.get_path('downfile')
    down_data = np.loadtxt(downfile)
    swc_path = description.manifest.get_path('MORPHOLOGY')
    #args, up_data, down_data, swc_path = process_inputs(parser)

    data = neuron_passive_fit(up_data, down_data, swc_path, limit)
    output_file = description.manifest.get_path('fit_1_file')
    #limit = description.data['runs'][0]['passive_fit_limit']

    json_utilities.write(output_file, data)

if __name__ == "__main__": main()

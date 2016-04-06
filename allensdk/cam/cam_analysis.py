# Copyright 2016 Allen Institute for Brain Science
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

from allensdk.cam.static_grating import StaticGrating
from allensdk.cam.locally_sparse_noise import LocallySparseNoise
from allensdk.cam.natural_scenes import NaturalScenes
from allensdk.cam.drifting_grating import DriftingGrating
from allensdk.cam.natural_movie import NaturalMovie

from allensdk.core.cam_nwb_data_set import CamNwbDataSet

import allensdk.cam.cam_plotting as cp
import argparse, logging

class CamAnalysis(object):
    _log = logging.getLogger('allensdk.cam.cam_analysis')    
    STIMULUS_A = 'A'
    STIMULUS_B = 'B'
    STIMULUS_C = 'C'

    def __init__(self, nwb_path, save_path, meta_data=None):
        self.nwb = CamNwbDataSet(nwb_path)                        
        self.save_path = save_path

        if meta_data is None:
            meta_data = {}

        self.meta_data = self.nwb.get_meta_data()
        for k,v in meta_data.iteritems():
            self.meta_data[k] = v

    def append_meta_data(self, df):
        for k,v in self.meta_data.iteritems():
            df[k] = v

    def save_stimulus_a(self, dg, nm1, nm3, peak):
        nwb = CamNwbDataSet(self.save_path)
        nwb.save_analysis_dataframes(
            ('stim_table_dg', dg.stim_table),
            ('sweep_response_dg', dg.sweep_response),
            ('mean_sweep_response_dg', dg.mean_sweep_response),
            ('peak', peak),        
            ('sweep_response_nm1', nm1.sweep_response),
            ('stim_table_nm1', nm1.stim_table),
            ('sweep_response_nm3', nm3.sweep_response))
        
        nwb.save_analysis_arrays(
            ('celltraces_dff', nm1.celltraces_dff),
            ('response_dg', dg.response),
            ('binned_cells_sp', nm1.binned_cells_sp),
            ('binned_cells_vis', nm1.binned_cells_vis),
            ('binned_dx_sp', nm1.binned_dx_sp),
            ('binned_dx_vis', nm1.binned_dx_vis))
    
        
    def save_stimulus_b(self, sg, nm1, ns, peak): 
        nwb = CamNwbDataSet(self.save_path)
        nwb.save_analysis_dataframes(
            ('stim_table_sg', sg.stim_table),
            ('sweep_response_sg', sg.sweep_response),
            ('mean_sweep_response_sg', sg.mean_sweep_response),
            ('sweep_response_nm1', nm1.sweep_response),
            ('stim_table_nm1', nm1.stim_table),
            ('sweep_response_ns', ns.sweep_response),
            ('stim_table_ns', ns.stim_table),
            ('mean_sweep_response_ns', ns.mean_sweep_response),
            ('peak', peak))

        nwb.save_analysis_arrays(
            ('celltraces_dff', nm1.celltraces_dff),
            ('response_sg', sg.response),
            ('response_ns', ns.response),
            ('binned_cells_sp', nm1.binned_cells_sp),
            ('binned_cells_vis', nm1.binned_cells_vis),
            ('binned_dx_sp', nm1.binned_dx_sp),
            ('binned_dx_vis', nm1.binned_dx_vis))
    
    
    def save_stimulus_c(self, lsn, nm1, nm2, peak):                
        nwb = CamNwbDataSet(self.save_path)
        nwb.save_analysis_dataframes(
            ('stim_table_lsn', lsn.stim_table),
            ('sweep_response_nm1', nm1.sweep_response),
            ('peak', peak),
            ('sweep_response_nm2', nm2.sweep_response),
            ('sweep_response_lsn', lsn.sweep_response),
            ('mean_sweep_response_lsn', lsn.mean_sweep_response))  
        
        nwb.save_analysis_arrays(
            ('receptive_field_lsn', lsn.receptive_field),
            ('celltraces_dff', nm1.celltraces_dff),
            ('binned_dx_sp', nm1.binned_dx_sp),
            ('binned_dx_vis', nm1.binned_dx_vis),    
            ('binned_cells_sp', nm1.binned_cells_sp),
            ('binned_cells_vis', nm1.binned_cells_vis))
    
    
    def stimulus_a(self, plot_flag=False, save_flag=True):
        dg = DriftingGrating(self)
        nm3 = NaturalMovie(self, 'natural_movie_three')    
        nm1 = NaturalMovie(self, 'natural_movie_one')        
        CamAnalysis._log.info("Stimulus A analyzed")
        peak = multi_dataframe_merge([nm1.peak_run, dg.peak, nm1.peak, nm3.peak])
        self.append_meta_data(peak)

        if plot_flag:
            cp.plot_3sa(dg, nm1, nm3)
            cp.plot_drifting_grating_traces()(dg)
    
        if save_flag:
            self.save_stimulus_a(dg, nm1, nm3, peak)
    
    def stimulus_b(self, plot_flag=False, save_flag=True):
        sg = StaticGrating(self)    
        ns = NaturalScenes(self)
        nm1 = NaturalMovie(self, 'natural_movie_one')            
        CamAnalysis._log.info("Stimulus B analyzed")
        peak = multi_dataframe_merge([nm1.peak_run, sg.peak, ns.peak, nm1.peak])
        self.append_meta_data(peak)
                
        if plot_flag:
            cp.plot_3sb(sg, nm1, ns)
            cp.plot_ns_traces(ns)
            cp.plot_sg_traces(sg)
                    
        if save_flag:
            self.save_stimulus_b(sg, nm1, ns, peak)
    
    def stimulus_c(self, plot_flag=False, save_flag=True):
        nm2 = NaturalMovie(self, 'natural_movie_two')
        lsn = LocallySparseNoise(self)
        nm1 = NaturalScenes(self, 'natural_movie_one')
        CamAnalysis._log.info("Stimulus C analyzed")
        peak = multi_dataframe_merge([nm1.peak_run, nm1.peak, nm2.peak])
        self.append_meta_data(peak)
                
        if plot_flag:
            cp.plot_3sc(lsn, nm1, nm2)
            cp.plot_lsn_traces(lsn)
    
        if save_flag:
            self.save_stimulus_c(lsn, nm1, nm2, peak)

def multi_dataframe_merge(dfs):
    out_df = None
    for i,df in enumerate(dfs):
        if out_df is None:
            out_df = df
        else:
            out_df = out_df.merge(df, left_index=True, right_index=True, suffixes=['','_%d' % i])
    return out_df
    
                    
def run_cam_analysis(stimulus, nwb_path, save_path, meta_data=None, plot_flag=False):
    cam_analysis = CamAnalysis(nwb_path, save_path, meta_data)

    if stimulus == CamAnalysis.STIMULUS_A:
        cam_analysis.stimulus_a(plot_flag)
    elif stimulus == CamAnalysis.STIMULUS_B:
        cam_analysis.stimulus_b(plot_flag)
    elif stimulus == CamAnalysis.STIMULUS_C:
        cam_analysis.stimulus_c(plot_flag)
    else:
        raise IndexError("Unknown stimulus: %s" % stimulus)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nwb", required=True)
    parser.add_argument("--output_nwb", default=None)

    # TODO: unhardcode
    parser.add_argument("--stimulus", default=CamAnalysis.STIMULUS_A)
    parser.add_argument("--plot", action='store_true')

    # meta data
    # TODO: remove
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--experiment_id", type=int, default=None)
    parser.add_argument("--area", type=str, default=None)

    args = parser.parse_args()

    if args.output_nwb is None:
        args.output_nwb = args.input_nwb

    meta_data = {}
    if args.experiment_id is not None:
        meta_data['experiment_id'] = args.experiment_id
    if args.area is not None:
        meta_data['area'] = args.area
    if args.depth is not None:
        meta_data['depth'] = args.depth

    run_cam_analysis(args.stimulus, args.input_nwb, args.output_nwb, meta_data, args.plot)


if __name__=='__main__': main()
    

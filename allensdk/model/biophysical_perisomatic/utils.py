# Copyright 2015 Allen Institute for Brain Science
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

import logging
from allensdk.model.biophys_sim.neuron.hoc_utils import HocUtils
from allensdk.core.nwb_data_set import NwbDataSet


class Utils(HocUtils):
    '''A helper class for NEURON functionality needed for
    perisomatic biophysical simulations.
    
    Attributes
    ----------
    h : object
        The NEURON hoc object.
    nrn : object
        The NEURON python object.
    neuron : module
        The NEURON module.
    '''
    
    _log = logging.getLogger(__name__)
    
    def __init__(self, description):
        super(Utils, self).__init__(description)
        self.stim = None
        self.stim_curr = None
        self.sampling_rate = None
    
        self.stim_vec_list = []
    
    
    def generate_morphology(self, morph_filename):
        '''Load a swc-format cell morphology file.
        
        Parameters
        ----------
        morph_filename : string
            Path to swc.
        '''
        h = self.h
        
        swc = self.h.Import3d_SWC_read()
        swc.input(morph_filename)
        imprt = self.h.Import3d_GUI(swc, 0)
        
        h("objref this")
        imprt.instantiate(h.this)
        
        h("soma[0] area(0.5)")
        for sec in h.allsec():
            sec.nseg = 1 + 2 * int(sec.L / 40)
            if sec.name()[:4] == "axon":
                h.delete_section(sec=sec)
        h('create axon[2]')
        for sec in h.axon:
            sec.L = 30
            sec.diam = 1
            sec.nseg = 1 + 2 * int(sec.L / 40)
        h.axon[0].connect(h.soma[0], 0.5, 0)
        h.axon[1].connect(h.axon[0], 1, 0)
        
        h.define_shape()
    
    
    def load_cell_parameters(self):
        '''Configure a neuron after the cell morphology has been loaded.'''
        passive = self.description.data['passive'][0]
        genome = self.description.data['genome']
        conditions = self.description.data['conditions'][0]
        h = self.h
        
        h("access soma")
        
        # Set fixed passive properties
        for sec in h.allsec():
            sec.Ra = passive['ra']
            sec.insert('pas')
            for seg in sec:
                seg.pas.e = passive["e_pas"]
        
        for c in passive["cm"]:
            h('forsec "' + c["section"] + '" { cm = %g }' % c["cm"])
        
        # Insert channels and set parameters
        for p in genome:
            if p["section"] == "glob": # global parameter
                h(p["name"] + " = %g " % p["value"])
            else:
                if p["mechanism"] != "":
                    h('forsec "' + p["section"] + '" { insert ' + p["mechanism"] + ' }')
                h('forsec "' + p["section"] + '" { ' + p["name"] + ' = %g }' % p["value"])
        
        # Set reversal potentials
        for erev in conditions['erev']:
            h('forsec "' + erev["section"] + '" { ek = %g }' % erev["ek"])
            h('forsec "' + erev["section"] + '" { ena = %g }' % erev["ena"])
    
    
    def setup_iclamp(self,
                     stimulus_path,
                     sweep=0):
        '''Assign a current waveform as input stimulus.
        
        Parameters
        ----------
        stimulus_path : string
            NWB file name
        '''
        self.stim = self.h.IClamp(self.h.soma[0](0.5))  # TODO: does soma have to be parametrized?
        self.stim.amp = 0
        self.stim.delay = 0
        self.stim.dur = 1e12 # just set to be really big; doesn't need to match the waveform
        
        self.read_stimulus(stimulus_path, sweep=sweep)
        self.h.dt = self.sampling_rate
        stim_vec = self.h.Vector(self.stim_curr)
        stim_vec.play(self.stim._ref_amp, self.sampling_rate)
        
        stimulus_stop_index = len(self.stim_curr) - 1
        self.h.tstop = stimulus_stop_index * self.sampling_rate
        self.stim_vec_list.append(stim_vec)
    
    
    def read_stimulus(self, stimulus_path, sweep=0):
        '''load current values for a specific experiment sweep.
        
        Parameters
        ----------
        stimulus path : string
            NWB file name
        sweep : integer, optional
            sweep index
        '''
        Utils._log.info("reading stimulus path: %s, sweep %s" %
                        (stimulus_path, sweep))
        stimulus_data = NwbDataSet(stimulus_path)
        sweep_data = stimulus_data.get_sweep(sweep)
        self.stim_curr = sweep_data['stimulus'] * 1.0e9 # convert to nA for NEURON
        self.sampling_rate = 1.0e3 / sweep_data['sampling_rate'] # convert from Hz
    
    
    def record_values(self):
        '''Set up output voltage recording.'''
        vec = { "v": self.h.Vector(),
                "t": self.h.Vector() }
    
        vec["v"].record(self.h.soma[0](0.5)._ref_v)
        vec["t"].record(self.h._ref_t)
    
        return vec

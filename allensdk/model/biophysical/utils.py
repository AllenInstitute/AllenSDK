# Copyright 2015-2016 Allen Institute for Brain Science
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
import os
from ..biophys_sim.neuron.hoc_utils import HocUtils
from allensdk.core.nwb_data_set import NwbDataSet

PERISOMATIC_TYPE = "Biophysical - perisomatic"
ALL_ACTIVE_TYPE = "Biophysical - all active"


def create_utils(description, model_type=None):
    if model_type is None:
        model_type = PERISOMATIC_TYPE

    if model_type == PERISOMATIC_TYPE:
        return Utils(description)
    elif model_type == ALL_ACTIVE_TYPE:
        return AllActiveUtils(description)


class Utils(HocUtils):
    '''A helper class for NEURON functionality needed for
    biophysical simulations.

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
            sec.nseg = 1 + 2 * int(sec.L / 40.0)
            if sec.name()[:4] == "axon":
                h.delete_section(sec=sec)
        h('create axon[2]')
        for sec in h.axon:
            sec.L = 30
            sec.diam = 1
            sec.nseg = 1 + 2 * int(sec.L / 40.0)
        h.axon[0].connect(h.soma[0], 0.5, 0.0)
        h.axon[1].connect(h.axon[0], 1.0, 0.0)

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
            if p["section"] == "glob":  # global parameter
                h(p["name"] + " = %g " % p["value"])
            else:
                if p["mechanism"] != "":
                    h('forsec "' + p["section"] +
                      '" { insert ' + p["mechanism"] + ' }')
                h('forsec "' + p["section"] +
                  '" { ' + p["name"] + ' = %g }' % p["value"])

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
        self.stim = self.h.IClamp(self.h.soma[0](0.5))
        self.stim.amp = 0
        self.stim.delay = 0
        # just set to be really big; doesn't need to match the waveform
        self.stim.dur = 1e12

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
        Utils._log.info(
            "reading stimulus path: %s, sweep %s",
            stimulus_path,
            sweep)

        stimulus_data = NwbDataSet(stimulus_path)
        sweep_data = stimulus_data.get_sweep(sweep)

        # convert to nA for NEURON
        self.stim_curr = sweep_data['stimulus'] * 1.0e9

        # convert from Hz
        self.sampling_rate = 1.0e3 / sweep_data['sampling_rate']

    def record_values(self):
        '''Set up output voltage recording.'''
        vec = {"v": self.h.Vector(),
               "t": self.h.Vector()}

        vec["v"].record(self.h.soma[0](0.5)._ref_v)
        vec["t"].record(self.h._ref_t)

        return vec


class AllActiveUtils(Utils):

    def generate_morphology(self, morph_filename):
        '''Load a neurolucida or swc-format cell morphology file.

        Parameters
        ----------
        morph_filename : string
            Path to morphology.
        '''

        morph_basename = os.path.basename(morph_filename)
        morph_extension = morph_basename.split('.')[-1]
        if morph_extension.lower() == 'swc':
            morph = self.h.Import3d_SWC_read()
        elif morph_extension.lower() == 'asc':
            morph = self.h.Import3d_Neurolucida3()
        else:
            raise Exception("Unknown filetype: %s" % morph_extension)

        morph.input(morph_filename)
        imprt = self.h.Import3d_GUI(morph, 0)

        self.h("objref this")
        imprt.instantiate(self.h.this)

        for sec in self.h.allsec():
            sec.nseg = 1 + 2 * int(sec.L / 40.0)

        self.h("soma[0] area(0.5)")
        axon_diams = [self.h.axon[0].diam, self.h.axon[0].diam]
        self.h.distance(sec=self.h.soma[0])
        for sec in self.h.allsec():
            if sec.name()[:4] == "axon":
                if self.h.distance(0.5, sec=sec) > 60:
                    axon_diams[1] = sec.diam
                    break
        for sec in self.h.allsec():
            if sec.name()[:4] == "axon":
                self.h.delete_section(sec=sec)
        self.h('create axon[2]')
        for index, sec in enumerate(self.h.axon):
            sec.L = 30
            sec.diam = axon_diams[index]

        for sec in self.h.allsec():
            sec.nseg = 1 + 2 * int(sec.L / 40.0)

        self.h.axon[0].connect(self.h.soma[0], 1.0, 0.0)
        self.h.axon[1].connect(self.h.axon[0], 1.0, 0.0)

        # make sure diam reflects 3d points
        self.h.area(.5, sec=self.h.soma[0])

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
            # for seg in sec:
            #     seg.pas.e = passive["e_pas"]

        # for c in passive["cm"]:
        #     h('forsec "' + c["section"] + '" { cm = %g }' % c["cm"])

        # Insert channels and set parameters
        for p in genome:
            section_array = p["section"]
            mechanism = p["mechanism"]
            param_name = p["name"]
            param_value = float(p["value"])
            if section_array == "glob":  # global parameter
                h(p["name"] + " = %g " % p["value"])
            else:
                if hasattr(h, section_array):
                    if mechanism != "":
                        print('Adding mechanism %s to %s'
                              % (mechanism, section_array))
                        for section in getattr(h, section_array):
                            if self.h.ismembrane(str(mechanism),
                                                 sec=section) != 1:
                                section.insert(mechanism)

                    print('Setting %s to %.6g in %s'
                          % (param_name, param_value, section_array))
                    for section in getattr(h, section_array):
                        setattr(section, param_name, param_value)

        # Set reversal potentials
        for erev in conditions['erev']:
            erev_section_array = erev["section"]
            ek = float(erev["ek"])
            ena = float(erev["ena"])

            print('Setting ek to %.6g and ena to %.6g in %s'
                  % (ek, ena, erev_section_array))

            if hasattr(h, erev_section_array):
                for section in getattr(h, erev_section_array):
                    if self.h.ismembrane("k_ion", sec=section) == 1:
                        setattr(section, 'ek', ek)

                    if self.h.ismembrane("na_ion", sec=section) == 1:
                        setattr(section, 'ena', ena)
            else:
                print("Warning: can't set erev for %s, "
                      "section array doesn't exist" % erev_section_array)

        self.h.v_init = conditions['v_init']
        self.h.celsius = conditions['celsius']

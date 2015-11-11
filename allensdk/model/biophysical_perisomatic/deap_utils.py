from allensdk.model.biophys_sim.neuron.hoc_utils import HocUtils
from allensdk.ephys.feature_extractor import EphysFeatureExtractor
import logging

import numpy as np

class Utils(HocUtils):
    _log = logging.getLogger(__name__)

    def __init__(self, description):
        super(Utils, self).__init__(description)
        self.stim = None
        self.stim_curr = None
        self.sampling_rate = None
        self.cell = self.h.cell()

    def generate_morphology(self, morph_filename):
        h = self.h
        cell = self.cell

        swc = self.h.Import3d_SWC_read()
        swc.input(morph_filename)
        imprt = self.h.Import3d_GUI(swc, 0)
        imprt.instantiate(cell)

        for seg in cell.soma[0]:
            seg.area()

        for sec in cell.all:
            sec.nseg = 1 + 2 * int(sec.L / 40)

        cell.simplify_axon()
        for sec in cell.axonal:
            sec.L = 30
            sec.diam = 1
            sec.nseg = 1 + 2 * int(sec.L / 40)
        cell.axon[0].connect(cell.soma[0], 0.5, 0)
        cell.axon[1].connect(cell.axon[0], 1, 0)
        h.define_shape()

    def load_cell_parameters(self):
        cell = self.cell
        passive = self.description.data['passive'][0]
        conditions = self.description.data['conditions'][0]
        channels = self.description.data['channels']
        addl_params = self.description.data['addl_params']

        # Set passive properties
        for sec in cell.all:
            sec.Ra = passive['ra']
            sec.cm = passive['cm'][sec.name().split(".")[1][:4]]
            sec.insert('pas')
            for seg in sec:
                seg.pas.e = passive["e_pas"]
        self.h.v_init = passive["e_pas"]

        # Insert channels and set parameters
        for c in channels:
            if c["mechanism"] != "":
                sections = [s for s in cell.all if s.name().split(".")[1][:4] == c["section"]]
                for sec in sections:
                    sec.insert(c["mechanism"])

        for ap in addl_params:
            if ap["mechanism"] != "":
                sections = [s for s in cell.all if s.name().split(".")[1][:4] == ap["section"]]
                for sec in sections:
                    sec.insert(ap["mechanism"])

        # Set reversal potentials
        for erev in conditions['erev']:
            sections = [s for s in cell.all if s.name().split(".")[1][:4] == erev["section"]]
            for sec in sections:
                sec.ena = erev["ena"]
                sec.ek = erev["ek"]

    def set_normalized_parameters(self, params):
        channels_and_others = self.description.data['channels'] + self.description.data['addl_params']
        for i, p in enumerate(params):
            c = channels_and_others[i]
            value = p * (c["max"] - c["min"]) + c["min"]
            sections = [s for s in self.cell.all if s.name().split(".")[1][:4] == c["section"]]
            for sec in sections:
                param_name = c["parameter"]
                if c["mechanism"] != "":
                    param_name += "_" + c["mechanism"]
                setattr(sec, param_name, value)

    def set_actual_parameters(self, params):
        channels_and_others = self.description.data['channels'] + self.description.data['addl_params']
        for i, p in enumerate(params):
            c = channels_and_others[i]
            sections = [s for s in self.cell.all if s.name().split(".")[1][:4] == c["section"]]
            for sec in sections:
                param_name = c["parameter"]
                if c["mechanism"] != "":
                    param_name += "_" + c["mechanism"]
                setattr(sec, param_name, p)

    def normalize_actual_parameters(self, params):
        params_array = np.array(params)
        channels_and_others = self.description.data['channels'] + self.description.data['addl_params']
        max_vals = np.array([c["max"] for c in channels_and_others])
        min_vals = np.array([c["min"] for c in channels_and_others])

        normalized_params = (params_array - min_vals) / (max_vals - min_vals)
        return normalized_params.tolist()

    def actual_parameters_from_normalized(self, params):
        actual_params = []
        channels_and_others = self.description.data['channels'] + self.description.data['addl_params']
        for i, p in enumerate(params):
            c = channels_and_others[i]
            value = p * (c["max"] - c["min"]) + c["min"]
            actual_params.append(value)
        return actual_params

    def insert_iclamp(self):
        self.stim = self.h.IClamp(self.cell.soma[0](0.5))

    def set_iclamp_params(self, amp, delay, dur):
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

    def calculate_feature_errors(self, t_ms, v, i):
        # Special case checks and penalty values
        minimum_num_spikes = 2
        missing_penalty_value = 20.0
        max_fail_penalty = 250.0
        min_fail_penalty = 75.0
        overkill_reduction = 0.75
        variance_factor = 0.1

        fail_trace = False

        delay = self.stim.delay * 1e-3
        duration = self.stim.dur * 1e-3
        t = t_ms * 1e-3
        feature_names = self.description.data['features']

        # penalize for failing to return to rest
        start_index = np.flatnonzero(t >= delay)[0]
        if np.abs(v[-1] - v[:start_index].mean()) > 2.0:
            fail_trace = True
        else:
            prefeat_fx = EphysFeatureExtractor()
            prefeat_fx.process_instance("", v, i, t, 0, delay, "")
            if prefeat_fx.feature_list[0].mean['n_spikes'] > 0:
                fail_trace = True

        if not fail_trace:
            fx = EphysFeatureExtractor()
            fx.process_instance("", v, i, t, delay, duration, "")
            if fx.feature_list[0].mean['n_spikes'] < minimum_num_spikes:
                fail_trace = True

        if fail_trace:
            variance_start = np.flatnonzero(t >= delay - 0.1)[0]
            variance_end = np.flatnonzero(t >= (delay + duration) / 2.0)[0]
            trace_variance = v[variance_start:variance_end].var()
            error_value = max(max_fail_penalty - trace_variance * variance_factor, min_fail_penalty)
            errs = np.ones(len(feature_names)) * error_value
        else:
            target_features = self.description.data['target_features']
            target_features_dict = {f["name"]: {"mean": f["mean"], "stdev": f["stdev"]} for f in target_features}
            errs = []
            for f in feature_names:
                if f in fx.feature_list[0].mean and fx.feature_list[0].mean[f] is not None:
                    target_mean = target_features_dict[f]['mean']
                    target_stdev = target_features_dict[f]['stdev']
                    model_mean = fx.feature_list[0].mean[f]
                    errs.append(np.abs((model_mean - target_mean) / target_stdev))
                else:
                    errs.append(missing_penalty_value)
            errs = np.array(errs)
        return errs

    def record_values(self):
        v_vec = self.h.Vector()
        t_vec = self.h.Vector()
        i_vec = self.h.Vector()

        v_vec.record(self.cell.soma[0](0.5)._ref_v)
        i_vec.record(self.stim._ref_amp)
        t_vec.record(self.h._ref_t)

        return v_vec, i_vec, t_vec

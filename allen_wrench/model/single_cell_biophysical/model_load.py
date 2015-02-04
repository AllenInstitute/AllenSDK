from neuron import h
from load_cell_parameters import load_cell_parameters
import json


def load_model_from_lims(morphology_path,
                         fit_parameter_json):
    h.load_file("stdgui.hoc")
    h.load_file("import3d.hoc")

    with open(fit_parameter_json, 'r') as f:
        params = json.load(f)

    generate_morphology(morphology_path.encode('ascii', 'ignore'))
    load_cell_parameters(h, params)
    setup_conditions(params)


def generate_morphology(morph_filename):
    swc = h.Import3d_SWC_read()
    swc.input(morph_filename)
    imprt = h.Import3d_GUI(swc, 0)
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


def setup_conditions(params):
    h.celsius = params["conditions"]["celsius"]
    h.v_init = params["conditions"]["v_init"]


def record_values():
    vec = { "v": h.Vector(),
            "t": h.Vector() }

    vec["v"].record(h.soma[0](0.5)._ref_v)
    vec["t"].record(h._ref_t)

    return vec

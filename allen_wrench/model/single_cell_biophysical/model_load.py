from neuron import h
import json


def load_model_from_lims(morphology_path,
                         fit_parameter_json):
    h.load_file("stdgui.hoc")
    h.load_file("import3d.hoc")

    with open(fit_parameter_json, 'r') as f:
        params = json.load(f)

    generate_morphology(morphology_path.encode('ascii', 'ignore'))
    load_cell_parameters(params)
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


def load_cell_parameters(params):
    h("access soma")

    # Set fixed passive properties
    for sec in h.allsec():
        sec.Ra = 100
        sec.cm = 1
        sec.insert('pas')
        for seg in sec:
            seg.pas.e = params["passive"]["e_pas"]
    h('forsec "apic" { cm = ' + params["passive"]["apic_cm"] + '}')
    h('forsec "dend" { cm = ' + params["passive"]["dend_cm"] + '}')

    # Insert channels and set parameters
    for p in params["genome"]:
        if p["section"] == "glob": # global parameter
            h(p["name"] + " = %g " % p["value"])
        else:
            if p["mechanism"] != "":
                h('forsec "' + p["section"] + '" { insert ' + p["mechanism"] + ' }')
            h('forsec "' + p["section"] + '" { ' + p["name"] + ' = %g }' % p["value"])

    # Set reversal potentials - some of these will give errors, but not sure how to check if a given ion is used in a given section
    h('forsec "soma" { ek = ' + params["conditions"]["ek"] + ' }')
    h('forsec "axon" { ek = ' + params["conditions"]["ek"] + ' }')
    h('forsec "dend" { ek = ' + params["conditions"]["ek"] + ' }')
    h('forsec "apic" { ek = ' + params["conditions"]["ek"] + ' }')
    h('forsec "soma" { ena = ' + params["conditions"]["ena"] + ' }')
    h('forsec "axon" { ena = ' + params["conditions"]["ena"] + ' }')
    h('forsec "dend" { ena = ' + params["conditions"]["ena"] + ' }')
    h('forsec "apic" { ena = ' + params["conditions"]["ena"] + ' }')


def setup_conditions(params):
    h.dt = 0.005
    h.celsius = params["conditions"]["celsius"]
    h.v_init = params["conditions"]["v_init"]


def track_typical_values():
    vec = {}
    for n in ["v", "v_ax0", "v_ax1", "t", "ina_t", "ina", "ica", "ik", "ikp", "ikt", "ik3", "hna", "mna", "gna"]:
        vec[n] = h.Vector()

    vec["v"].record(h.soma[0](0.5)._ref_v)
    vec["t"].record(h._ref_t)
    vec["ina"].record(h.soma[0](0.5)._ref_ina)
    vec["ica"].record(h.soma[0](0.5)._ref_ica)
    vec["ik"].record(h.soma[0](0.5)._ref_ik)

    return vec

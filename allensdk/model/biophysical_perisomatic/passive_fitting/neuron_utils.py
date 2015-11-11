from neuron import h
import sys, os
from cStringIO import StringIO
from output_grabber import OutputGrabber

def load_morphology(filename):
    swc = h.Import3d_SWC_read()
    swc.input(str(filename))
    imprt = h.Import3d_GUI(swc, 0)
    h("objref this")
    imprt.instantiate(h.this)


def parse_neuron_output(output_str):
    printed_fields = {}

    for line in output_str.split('\n'):
        if line.startswith('nquad'):
            continue
        toks = line.split()
        if len(toks) == 2:
            v = toks[1].strip()
            try:
                v = float(v)
            except:
                pass
            
            printed_fields[toks[0].strip()] = v

    return printed_fields


def read_neuron_fit_stdout(func):
    def call(*args, **kwargs):

        g = OutputGrabber()
        g.start()
        data = func(*args, **kwargs)
        g.stop()

        printed_fields = parse_neuron_output(g.capturedtext)
        data.update(printed_fields)

        return data

    return call



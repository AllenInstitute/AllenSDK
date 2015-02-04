def load_cell_parameters(h, params):
    h("access soma")

    # Set fixed passive properties
    for sec in h.allsec():
        sec.Ra = params['passive']['ra']
        sec.insert('pas')
        for seg in sec:
            seg.pas.e = params["passive"]["e_pas"]

    for c in params["passive"]["cm"]:
        h('forsec "' + c["section"] + '" { cm = %g }' % c["cm"])

    # Insert channels and set parameters
    for p in params["genome"]:
        if p["section"] == "glob": # global parameter
            h(p["name"] + " = %g " % p["value"])
        else:
            if p["mechanism"] != "":
                h('forsec "' + p["section"] + '" { insert ' + p["mechanism"] + ' }')
            h('forsec "' + p["section"] + '" { ' + p["name"] + ' = %g }' % p["value"])

    # Set reversal potentials
    for erev in params['conditions']['erev']:
        h('forsec "' + erev["section"] + '" { ek = %g }' % erev["ek"])
        h('forsec "' + erev["section"] + '" { ena = %g }' % erev["ena"])


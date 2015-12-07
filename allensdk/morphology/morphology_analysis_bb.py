from neurom import ezy
import neurom.analysis.morphtree as mt
from neurom.analysis import morphmath as mm
from neurom.core import tree as tr
from neurom.core.dataformat import COLS
from neurom.core.types import TreeType
from morphology_analysis_bb import *
import numpy as np
import math
import sys

AXON = 0
BASAL = 1
APICAL = 2
DENDRITE = 3

def get_number_of_nodes(nrn):
    num_ax = nrn.get_n_sections(TreeType.axon)
    num_ba = nrn.get_n_sections(TreeType.basal_dendrite)
    num_ap = nrn.get_n_sections(TreeType.apical_dendrite)
    return [ num_ax, num_ba, num_ap, (num_ba+num_ap) ]
    
def get_soma_surface_area(nrn):
    return nrn.get_soma_surface_area()

def get_number_of_stems(nrn):
    trees = nrn.neurites
    num_ax = 0
    num_ba = 0
    num_ap = 0
    for i in range(len(trees)):
        tree = trees[i]
        tp = tree.value[COLS.TYPE]
        if tp == 2:
            num_ax += 1
        elif tp == 3:
            num_ba += 1
        elif tp == 4:
            num_ap += 1
        elif tp > 4:
            print("Unexpected stem type (%d) -- ignoring it" % tp)
    return [ num_ax, num_ba, num_ap, (num_ba+num_ap) ]

def get_number_of_bifurcations(nrn):
    trees = nrn.neurites
    cnt_ax = 0
    cnt_ba = 0
    cnt_ap = 0
    num_ax = 0
    num_ba = 0
    num_ap = 0
    for i in range(len(trees)):
        tree = trees[i]
        tp = tree.value[COLS.TYPE]
        if tp == 2:
            cnt_ax += 1
            num_ax += mt.n_bifurcations(tree)
        elif tp == 3:
            cnt_ba += 1
            num_ba += mt.n_bifurcations(tree)
        elif tp == 4:
            cnt_ap += 1
            num_ap += mt.n_bifurcations(tree)
    if cnt_ax == 0:
        num_ax = float('nan')
    if cnt_ba == 0:
        num_ba = float('nan')
    if cnt_ap == 0:
        num_ap = float('nan')
    # average dendrite diameter, or NaN if no dendrite
    if cnt_ap == 0:
        num_de = num_ba
    elif cnt_ba == 0:
        num_de = num_ap
    else:
        num_de = num_ba + num_ap
    return [ num_ax, num_ba, num_ap, num_de ]

def get_number_of_branches(nrn):
    axon = nrn.get_n_sections_per_neurite(TreeType.axon)
    if len(axon) > 0:
        axon_med = np.median(axon)
        axon_mean = np.mean(axon)
        axon_pk = max(axon)
        axon_tot = np.sum(axon)
    else:
        axon_med = float('nan')
        axon_mean = float('nan')
        axon_pk = float('nan')
        axon_tot = float('nan')
    basal = nrn.get_n_sections_per_neurite(TreeType.basal_dendrite)
    if len(basal) > 0:
        basal_med = np.median(basal)
        basal_mean = np.mean(basal)
        basal_pk = max(basal)
        basal_tot = np.sum(basal)
    else:
        basal_med = float('nan')
        basal_mean = float('nan')
        basal_pk = float('nan')
        basal_tot = float('nan')
    apical = nrn.get_n_sections_per_neurite(TreeType.apical_dendrite)
    if len(apical) > 0:
        apical_med = np.median(apical)
        apical_mean = np.mean(apical)
        apical_pk = max(apical)
        apical_tot = np.sum(apical)
    else:
        apical_med = float('nan')
        apical_mean = float('nan')
        apical_pk = float('nan')
        apical_tot = float('nan')
    # combined apical and basal
    if len(apical) == 0:
        dend_med = basal_med
        dend_mean = basal_mean
        dend_pk = basal_pk
        dend_tot = basal_tot
    elif len(basal) == 0:
        dend_med = apical_med
        dend_mean = apical_mean
        dend_pk = apical_pk
        dend_tot = apical_tot
    else:
        dend_med = np.median(np.concatenate((basal, apical)))
        dend_mean = np.mean(np.concatenate((basal, apical)))
        dend_pk = max(np.concatenate((basal, apical)))
        dend_tot = np.sum(np.concatenate((basal, apical)))
    #
    med = [ axon_med, basal_med, apical_med, dend_med ]
    mean = [ axon_mean, basal_mean, apical_mean, dend_mean ]
    pk = [ axon_pk, basal_pk, apical_pk, dend_pk ]
    tot = [ axon_tot, basal_tot, apical_tot, dend_tot ]
    return [ med, mean, pk, tot ]


def get_number_of_tips(nrn):
    trees = nrn.neurites
    num_ax = 0
    num_ba = 0
    num_ap = 0
    for i in range(len(trees)):
        tree = trees[i]
        tp = tree.value[COLS.TYPE]
        if tp == 2:
            num_ax += mt.n_terminations(tree)
        elif tp == 3:
            num_ba += mt.n_terminations(tree)
        elif tp == 4:
            num_ap += mt.n_terminations(tree)
    return [ num_ax, num_ba, num_ap, (num_ba+num_ap) ]

#########################################################################
# diameter

# internal function
def update_bounds(bounds, tree):
    mn, mx = mt.get_bounding_box(tree)
    bounds[0] = min(bounds[0], mn[0])
    bounds[1] = min(bounds[1], mn[1])
    bounds[2] = min(bounds[2], mn[2])
    bounds[3] = max(bounds[3], mx[0])
    bounds[4] = max(bounds[4], mx[1])
    bounds[5] = max(bounds[5], mx[2])

LOW_VAL = -1.0e100
BIG_VAL = 1.0e100

# returns [xmin, ymin, zmin, xmax, ymax, zmax]
def get_bounding_box_by_type(nrn, tp):
    global LOW_VAL, BIG_VAL
    trees = nrn.neurites
    bounds = np.zeros(6)
    bounds[0] = BIG_VAL
    bounds[1] = BIG_VAL
    bounds[2] = BIG_VAL
    bounds[3] = LOW_VAL
    bounds[4] = LOW_VAL
    bounds[5] = LOW_VAL
    for i in range(len(trees)):
        tree = trees[i]
        if tree.value[COLS.TYPE] == tp:
            update_bounds(bounds, tree)
    return bounds

def get_bounding_box(nrn):
    global LOW_VAL, BIG_VAL
    ax = get_bounding_box_by_type(nrn, 2)
    ba = get_bounding_box_by_type(nrn, 3)
    ap = get_bounding_box_by_type(nrn, 4)
    dend = np.zeros(6)
    dend[0] = min(ba[0], ap[0])
    dend[1] = min(ba[1], ap[1])
    dend[2] = min(ba[2], ap[2])
    dend[3] = max(ba[3], ap[3])
    dend[4] = max(ba[4], ap[4])
    dend[5] = max(ba[5], ap[5])
    if min(ax) == LOW_VAL or max(ax) == BIG_VAL:
        for i in range(len(ax)):
            ax[i] = float('nan')
    if min(ba) == LOW_VAL or max(ba) == BIG_VAL:
        for i in range(len(ba)):
            ba[i] = float('nan')
    if min(ap) == LOW_VAL or max(ap) == BIG_VAL:
        for i in range(len(ap)):
            ap[i] = float('nan')
    if min(dend) == LOW_VAL or max(dend) == BIG_VAL:
        for i in range(len(dend)):
            dend[i] = float('nan')
    return [ ax, ba, ap, dend ]

#########################################################################
# diameter

def get_diameter(nrn):
    # axon diameter, or NaN if no axon found
    axon = [v for v in nrn.iter_segments(mm.segment_radius, TreeType.axon)]
    if len(axon) > 0:
        dia_ax = 2.0 * np.mean(axon)
    else:
        dia_ax = float('nan')
    # basal dendrite diameter, or NaN
    basal = [v for v in nrn.iter_segments(mm.segment_radius, TreeType.basal_dendrite)]
    if len(basal) > 0:
        dia_ba = 2.0 * np.mean(basal)
    else:
        dia_ba = float('nan')
    # apical dendrite diameter, or NaN
    apical = [v for v in nrn.iter_segments(mm.segment_radius, TreeType.apical_dendrite)]
    if len(apical) > 0:
        dia_ap = 2.0 * np.mean(apical)
    else:
        dia_ap = float('nan')
    # average dendrite diameter, or NaN if no dendrite
    if len(apical) == 0:
        dia_dend = dia_ba
    elif len(basal) == 0:
        dia_dend = dia_ap
    else:
        dia_dend = 2.0 * np.mean(np.concatenate((basal, apical)))
    #
    return [ dia_ax, dia_ba, dia_ap, dia_dend ]

def get_min_diameter(nrn):
    # axon diameter, or NaN if no axon found
    axon = [v for v in nrn.iter_segments(mm.segment_radius, TreeType.axon)]
    if len(axon) > 0:
        dia_ax = 2.0 * min(axon)
    else:
        dia_ax = float('nan')
    # basal dendrite diameter, or NaN
    basal = [v for v in nrn.iter_segments(mm.segment_radius, TreeType.basal_dendrite)]
    if len(basal) > 0:
        dia_ba = 2.0 * min(basal)
    else:
        dia_ba = float('nan')
    # apical dendrite diameter, or NaN
    apical = [v for v in nrn.iter_segments(mm.segment_radius, TreeType.apical_dendrite)]
    if len(apical) > 0:
        dia_ap = 2.0 * min(apical)
    else:
        dia_ap = float('nan')
    # average dendrite diameter, or NaN if no dendrite
    if len(apical) == 0:
        dia_dend = dia_ba
    elif len(basal) == 0:
        dia_dend = dia_ap
    else:
        dia_dend = 2.0 * min(np.concatenate((basal, apical)))
    #
    return [ dia_ax, dia_ba, dia_ap, dia_dend ]

def get_max_diameter(nrn):
    # axon diameter, or NaN if no axon found
    axon = [v for v in nrn.iter_segments(mm.segment_radius, TreeType.axon)]
    if len(axon) > 0:
        dia_ax = 2.0 * max(axon)
    else:
        dia_ax = float('nan')
    # basal dendrite diameter, or NaN
    basal = [v for v in nrn.iter_segments(mm.segment_radius, TreeType.basal_dendrite)]
    if len(basal) > 0:
        dia_ba = 2.0 * max(basal)
    else:
        dia_ba = float('nan')
    # apical dendrite diameter, or NaN
    apical = [v for v in nrn.iter_segments(mm.segment_radius, TreeType.apical_dendrite)]
    if len(apical) > 0:
        dia_ap = 2.0 * max(apical)
    else:
        dia_ap = float('nan')
    # average dendrite diameter, or NaN if no dendrite
    if len(apical) == 0:
        dia_dend = dia_ba
    elif len(basal) == 0:
        dia_dend = dia_ap
    else:
        dia_dend = 2.0 * max(np.concatenate((basal, apical)))
    #
    return [ dia_ax, dia_ba, dia_ap, dia_dend ]

#########################################################################
# length

def get_length(nrn):
    num_ax = np.sum([v for v in nrn.iter_segments(mm.segment_length, TreeType.axon)])
    num_ba = np.sum([v for v in nrn.iter_segments(mm.segment_length, TreeType.basal_dendrite)])
    num_ap = np.sum([v for v in nrn.iter_segments(mm.segment_length, TreeType.apical_dendrite)])
    return [ num_ax, num_ba, num_ap, (num_ba+num_ap) ]

#########################################################################
# surface

def get_surface(nrn):
    num_ax = np.sum([v for v in nrn.iter_segments(mm.segment_area, TreeType.axon)])
    num_ba = np.sum([v for v in nrn.iter_segments(mm.segment_area, TreeType.basal_dendrite)])
    num_ap = np.sum([v for v in nrn.iter_segments(mm.segment_area, TreeType.apical_dendrite)])
    return [ num_ax, num_ba, num_ap, (num_ba+num_ap) ]

#########################################################################
# volume

def get_volume(nrn):
    axon = [v for v in nrn.iter_segments(mm.segment_volume, TreeType.axon)]
    if len(axon) == 0:
        num_ax = float('nan')
    else:
        num_ax = np.sum(axon)
    basal = [v for v in nrn.iter_segments(mm.segment_volume, TreeType.basal_dendrite)]
    if len(basal) == 0:
        num_ba = float('nan')
    else:
        num_ba = np.sum(basal)
    apical = [v for v in nrn.iter_segments(mm.segment_volume, TreeType.apical_dendrite)]
    if len(apical) == 0:
        num_ap = float('nan')
    else:
        num_ap = np.sum(apical)
    # merge dendrite types
    if len(apical) == 0:
        dend = num_ba
    elif len(basal) == 0:
        dend = num_ap
    else:
        dend = num_ap + num_ba
    return [ num_ax, num_ba, num_ap, dend ]

#########################################################################
# max absolute distance from soma
def get_max_euclidean_distance(nrn):
    axon = [v for v in nrn.get_section_radial_distances(neurite_type=TreeType.axon)]
    if len(axon) > 0:
        num_ax = max(axon)
    else:
        num_ax = float('nan')
    basal = [v for v in nrn.get_section_radial_distances(neurite_type=TreeType.basal_dendrite)]
    if len(basal) > 0:
        num_ba = max(basal)
    else:
        num_ba = float('nan')
    apical = [v for v in nrn.get_section_radial_distances(neurite_type=TreeType.apical_dendrite)]
    if len(apical) > 0:
        num_ap = max(apical)
    else:
        num_ap = float('nan')
    # merge basal and apical
    if len(apical) == 0:
        dend = num_ba
    elif len(basal) == 0:
        dend = num_ap
    else:
        dend = max(num_ap, num_ba)
    return [ num_ax, num_ba, num_ap, dend ]

#########################################################################
# max path distance from soma

def get_max_path_distance(nrn):
    axon = [v for v in nrn.get_section_path_distances(neurite_type=TreeType.axon)]
    if len(axon) > 0:
        num_ax = max(axon)
    else:
        num_ax = float('nan')
    basal = [v for v in nrn.get_section_path_distances(neurite_type=TreeType.basal_dendrite)]
    if len(basal) > 0:
        num_ba = max(basal)
    else:
        num_ba = float('nan')
    apical = [v for v in nrn.get_section_path_distances(neurite_type=TreeType.apical_dendrite)]
    if len(apical) > 0:
        num_ap = max(apical)
    else:
        num_ap = float('nan')
    # merge basal and apical
    if len(apical) == 0:
        dend = num_ba
    elif len(basal) == 0:
        dend = num_ap
    else:
        dend = max(num_ap, num_ba)
    return [ num_ax, num_ba, num_ap, dend ]

#########################################################################
# max branch order

def get_max_branch_order(nrn):
    trees = nrn.neurites
    num_ax = 0
    num_ba = 0
    num_ap = 0
    for i in range(len(trees)):
        tree = trees[i]
        partitions = mt.partition(tree)
        if len(partitions) > 0:
            parts = max(partitions)
            tp = tree.value[COLS.TYPE]
            if tp == 2:
                num_ax = max(num_ax, parts)
            elif tp == 3:
                num_ba = max(num_ba, parts)
            elif tp == 4:
                num_ap = max(num_ap, parts)
    return [ num_ax, num_ba, num_ap, max(num_ba, num_ap) ]

#########################################################################
# sections per neurite

def get_trunk_diameter(nrn):
    axon = nrn.get_trunk_radii(TreeType.axon)
    if len(axon) > 0:
        axon_med = 2.0 * np.median(axon)
        axon_mean = 2.0 * np.mean(axon)
        axon_pk = 2.0 * max(axon)
    else:
        axon_med = float('nan')
        axon_mean = float('nan')
        axon_pk = float('nan')
    basal = nrn.get_trunk_radii(TreeType.basal_dendrite)
    if len(basal) > 0:
        basal_med = 2.0 * np.median(basal)
        basal_mean = 2.0 * np.mean(basal)
        basal_pk = 2.0 * max(basal)
    else:
        basal_med = float('nan')
        basal_mean = float('nan')
        basal_pk = float('nan')
    apical = nrn.get_trunk_radii(TreeType.apical_dendrite)
    if len(apical) > 0:
        apical_med = 2.0 * np.median(apical)
        apical_mean = 2.0 * np.mean(apical)
        apical_pk = 2.0 * max(apical)
    else:
        apical_med = float('nan')
        apical_mean = float('nan')
        apical_pk = float('nan')
    # combined apical and basal
    if len(apical) == 0:
        dend_med = basal_med
        dend_mean = basal_mean
        dend_pk = basal_pk
    elif len(basal) == 0:
        dend_med = apical_med
        dend_mean = apical_mean
        dend_pk = apical_pk
    else:
        dend_med = 2.0 * np.median(np.concatenate((basal, apical)))
        dend_mean = 2.0 * np.mean(np.concatenate((basal, apical)))
        dend_pk = 2.0 * max(np.concatenate((basal, apical)))
    #
    med = [ axon_med, basal_med, apical_med, dend_med ]
    mean = [ axon_mean, basal_mean, apical_mean, dend_mean ]
    pk = [ axon_pk, basal_pk, apical_pk, dend_pk ]
    return [ med, mean, pk ]

#########################################################################
# trunk length

def get_trunk_length(nrn):
    axon = nrn.get_trunk_lengths(TreeType.axon)
    if len(axon) > 0:
        axon_med = 2.0 * np.median(axon)
        axon_mean = 2.0 * np.mean(axon)
        axon_pk = 2.0 * max(axon)
    else:
        axon_med = float('nan')
        axon_mean = float('nan')
        axon_pk = float('nan')
    basal = nrn.get_trunk_lengths(TreeType.basal_dendrite)
    if len(basal) > 0:
        basal_med = 2.0 * np.median(basal)
        basal_mean = 2.0 * np.mean(basal)
        basal_pk = 2.0 * max(basal)
    else:
        basal_med = float('nan')
        basal_mean = float('nan')
        basal_pk = float('nan')
    apical = nrn.get_trunk_lengths(TreeType.apical_dendrite)
    if len(apical) > 0:
        apical_med = 2.0 * np.median(apical)
        apical_mean = 2.0 * np.mean(apical)
        apical_pk = 2.0 * max(apical)
    else:
        apical_med = float('nan')
        apical_mean = float('nan')
        apical_pk = float('nan')
    # combined apical and basal
    if len(apical) == 0:
        dend_med = basal_med
        dend_mean = basal_mean
        dend_pk = basal_pk
    elif len(basal) == 0:
        dend_med = apical_med
        dend_mean = apical_mean
        dend_pk = apical_pk
    else:
        dend_med = 2.0 * np.median(np.concatenate((basal, apical)))
        dend_mean = 2.0 * np.mean(np.concatenate((basal, apical)))
        dend_pk = 2.0 * max(np.concatenate((basal, apical)))
    #
    med = [ axon_med, basal_med, apical_med, dend_med ]
    mean = [ axon_mean, basal_mean, apical_mean, dend_mean ]
    pk = [ axon_pk, basal_pk, apical_pk, dend_pk ]
    return [ med, mean, pk ]

#########################################################################

def bifurcation_angle_local(nrn):
    axon = nrn.get_local_bifurcation_angles(TreeType.axon)
    if len(axon) > 0:
        num_ax = 180.0 * np.mean(axon) / math.pi
    else:
        num_ax = float('nan')
    basal = nrn.get_local_bifurcation_angles(TreeType.basal_dendrite)
    if len(basal) > 0:
        num_ba = 180.0 * np.mean(basal) / math.pi
    else:
        num_ba = float('nan')
    apical = nrn.get_local_bifurcation_angles(TreeType.apical_dendrite)
    if len(apical) > 0:
        num_ap = 180.0 * np.mean(apical) / math.pi
    else:
        num_ap = float('nan')
    # merge basal and apical
    if len(apical) == 0:
        dend = num_ba
    elif len(basal) == 0:
        dend = num_ax
    else:
        dend = 180.0 * np.mean(np.concatenate((apical, basal))) / math.pi
    return [ num_ax, num_ba, num_ap, dend ]

#########################################################################

def bifurcation_angle_remote(nrn):
    axon = nrn.get_remote_bifurcation_angles(TreeType.axon)
    if len(axon) > 0:
        num_ax = 180.0 * np.mean(axon) / math.pi
    else:
        num_ax = float('nan')
    basal = nrn.get_remote_bifurcation_angles(TreeType.basal_dendrite)
    if len(basal) > 0:
        num_ba = 180.0 * np.mean(basal) / math.pi
    else:
        num_ba = float('nan')
    apical = nrn.get_remote_bifurcation_angles(TreeType.apical_dendrite)
    if len(apical) > 0:
        num_ap = 180.0 * np.mean(apical) / math.pi
    else:
        num_ap = float('nan')
    # merge basal and apical
    if len(apical) == 0:
        dend = num_ba
    elif len(basal) == 0:
        dend = num_ax
    else:
        dend = 180.0 * np.mean(np.concatenate((apical, basal))) / math.pi
    return [ num_ax, num_ba, num_ap, dend ]

#########################################################################
#########################################################################

class Node(object):
    def __init__(self):
        self.n = -1
        self.t = -1
        self.x = -1
        self.y = -1
        self.z = -1
        self.r = -1
        self.pn = -1
        self.parent = None
        self.children = []

# BB requires that swc file be consecutive
# construct tmp file with any 'holes' in ID sequence removed
def make_swc_consecutive(swc_file):
    print("Making consecutive version of %s for BB librray" % swc_file)
    obj_list = []
    obj_hash = {}
    try:
        f = open(swc_file, "r")
        line = f.readline()
        while len(line) > 0:
            if not line.startswith('#'):
                toks = line.split(' ')
                obj = Node()
                obj.n = int(toks[0])
                obj.t = int(toks[1])
                obj.x = float(toks[2])
                obj.y = float(toks[3])
                obj.z = float(toks[4])
                obj.r = float(toks[5])
                obj.pn = int(toks[6].strip('\r'))
                obj_list.append(obj)
                obj_hash[obj.n] = obj
            line = f.readline()
        f.close()
    except IOError:
        print("Error opening '%d'", swc_file)
        sys.exit(1)
    # construct tree
    for j in range(len(obj_list)):
        obj = obj_list[j]
        if obj.pn >= 0:
            obj.parent = obj_hash[obj.pn]
            obj.parent.children.append(obj)
    # report number of parents w/ multiple children
    ctr = 0
    for i in range(len(obj_list)):
        obj = obj_list[i]
        if len(obj.children) > 1:
            ctr += 1
    # assign consecutive IDs to objects
    for j in range(len(obj_list)):
        obj_list[j].n = j+1
    # re-link objects with parents
    for j in range(len(obj_list)):
        obj = obj_list[j]
        if obj.pn >= 0:
            obj.pn = obj.parent.n
    # write out clean file
    tmp_name = swc_file + ".tmp.swc"
    try:
        f = open(tmp_name, "w")
        for j in range(len(obj_list)):
            obj = obj_list[j]
            f.write("%d %d %g %g %g %g %d\n" % (obj.n, obj.t, obj.x, obj.y, obj.z, obj.r, obj.pn))
        f.close()
    except IOError:
        print("Error creating '%d'", swc_file)
        sys.exit(1)
    return tmp_name



def record(feat, feat_desc, name, values):
    feat.append(values[AXON])
    feat.append(values[BASAL])
    feat.append(values[APICAL])
    feat.append(values[DENDRITE])
    feat_desc.append("bb_" + name % "axon")
    feat_desc.append("bb_" + name % "basal")
    feat_desc.append("bb_" + name % "apical")
    feat_desc.append("bb_" + name % "dendrite")

def compute_features(swc_file):
    try:
        nrn = ezy.load_neuron(swc_file)
    except:
        fname = make_swc_consecutive(swc_file)
        nrn = ezy.load_neuron(fname)

    feat = []
    desc = []

    ####################################################################
    feat.append(get_soma_surface_area(nrn))
    desc.append("bb_soma_surface")
    ####################################################################
    record(feat, desc, "number_%s_sections", get_number_of_nodes(nrn))
    record(feat, desc, "number_%s_stems", get_number_of_stems(nrn))
    record(feat, desc, "number_%s_bifurcations", get_number_of_bifurcations(nrn))
    #
    branches = get_number_of_branches(nrn)
    record(feat, desc, "median_%s_branches", branches[0])
    record(feat, desc, "mean_%s_branches", branches[1])
    record(feat, desc, "max_%s_branches", branches[2])
    record(feat, desc, "number_%s_branches", branches[3])
    #
    record(feat, desc, "number_%s_tips", get_number_of_tips(nrn))
    # bounding box
    bounds = get_bounding_box(nrn)
    width = np.zeros(4)
    height = np.zeros(4)
    depth = np.zeros(4)
    for i in range(4):
        width[i] = bounds[i][3] - bounds[i][0]
        height[i] = bounds[i][4] - bounds[i][1]
        depth[i] = bounds[i][5] - bounds[i][2]
    record(feat, desc, "overall_%s_width", width)
    record(feat, desc, "overall_%s_height", height)
    record(feat, desc, "overall_%s_depth", depth)
    #
    record(feat, desc, "mean_%s_diameter", get_diameter(nrn))
    record(feat, desc, "min_%s_diameter", get_min_diameter(nrn))
    record(feat, desc, "max_%s_diameter", get_max_diameter(nrn))
    record(feat, desc, "total_%s_length", get_length(nrn))
    record(feat, desc, "total_%s_surface", get_surface(nrn))
    record(feat, desc, "total_%s_volume", get_volume(nrn))
    record(feat, desc, "max_%s_euclidean_distance", get_max_euclidean_distance(nrn))
    record(feat, desc, "max_%s_path_distance", get_max_path_distance(nrn))
    record(feat, desc, "max_%s_branch_order", get_max_branch_order(nrn))
    #
    trunk = get_trunk_diameter(nrn)
    record(feat, desc, "median_%s_trunk_diameter", trunk[0])
    record(feat, desc, "mean_%s_trunk_diameter", trunk[1])
    record(feat, desc, "max_%s_trunk_diameter", trunk[2])
    #
    trunk = get_trunk_length(nrn)
    record(feat, desc, "median_%s_trunk_length", trunk[0])
    record(feat, desc, "mean_%s_trunk_length", trunk[1])
    record(feat, desc, "max_%s_trunk_length", trunk[2])
    #
    record(feat, desc, "%s_bifurcation_angle_local", bifurcation_angle_local(nrn))
    record(feat, desc, "%s_bifurcation_angle_remote", bifurcation_angle_remote(nrn))
    ####################################################################
    return feat, desc



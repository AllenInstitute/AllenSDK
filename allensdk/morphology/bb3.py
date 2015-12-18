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

#AXON = 0
#BASAL = 1
#APICAL = 2
#DENDRITE = 3
COL_X = COLS.X
COL_Y = COLS.Y
COL_Z = COLS.Z
COL_R = COLS.R
COL_T = COLS.TYPE
COL_N = COLS.ID
COL_P = COLS.P

TYPE_SOMA = 1
TYPE_AXON = 2
TYPE_BASAL = 3
TYPE_APICAL = 4
TYPE_DENDRITE = 10

AXIS_X = COL_X
AXIS_Y = COL_Y
AXIS_Z = COL_Z

def calculate_moments_on_axis(nrn, seg_type, axis):
    iterator = nrn.iter_neurites(tr.isegment)
    first = 0.0
    second = 0.0
    cnt = 0.0
    for seg in iterator:
        # see if this segment is part of search criteria
        st = seg[0].value[COL_T]
        if seg_type == TYPE_DENDRITE and (st==TYPE_BASAL or st==TYPE_APICAL):
            match = True
        elif st == seg_type:
            match = True
        else:
            match = False
        # if segment part of query, update moment calculation
        if match:
            sz = abs(seg[0].value[axis] - seg[1].value[axis])
            dist = abs(seg[0].value[axis] - nrn.soma.center[axis])
            pz = abs(seg[0].value[axis] + seg[1].value[axis]) / 2.0
            dist = abs(pz - nrn.soma.center[axis])
            #
            first += sz * dist
            second += sz * dist * dist
            cnt += 1.0
    if cnt > 0.0:
        first /= cnt
        second = second / cnt
    return first, second

def calculate_moments(nrn, seg_type):
    iterator = nrn.iter_neurites(tr.isegment)
    first = 0.0
    second = 0.0
    cnt = 0.0
    for seg in iterator:
        # see if this segment is part of search criteria
        st = seg[0].value[COL_T]
        if seg_type == TYPE_DENDRITE and (st==TYPE_BASAL or st==TYPE_APICAL):
            match = True
        elif st == seg_type:
            match = True
        else:
            match = False
        # if segment part of query, update moment calculation
        if match:
            dx = seg[0].value[COL_X] - seg[1].value[COL_X]
            dy = seg[0].value[COL_Y] - seg[1].value[COL_Y]
            dz = seg[0].value[COL_Z] - seg[1].value[COL_Z]
            sz = math.sqrt(dx*dx + dy*dy + dz*dz)
            px = (seg[0].value[COL_X] + seg[1].value[COL_X]) / 2.0
            py = (seg[0].value[COL_Y] + seg[1].value[COL_Y]) / 2.0
            pz = (seg[0].value[COL_Z] + seg[1].value[COL_Z]) / 2.0
            dist = np.linalg.norm(np.subtract(nrn.soma.center[0:3], (px,py,pz)))
            #
            first += sz * abs(dist)
            second += sz * dist * dist
            cnt += 1.0
    if cnt > 0.0:
        first /= cnt
        second = second / cnt
    return first, second

def compute_features(swc_file):
    nrn = ezy.load_neuron(swc_file)
    results = {}

    ####################################################################
    # number of neurites
    num_ba = nrn.get_n_neurites(TreeType.basal_dendrite)
    num_ap = nrn.get_n_neurites(TreeType.apical_dendrite)
    results["bb_number_neurites_apical"] = num_ap
    results["bb_number_neurites_basal"] = num_ba
    results["bb_number_neurites_dendrite"] = num_ap + num_ba

    ####################################################################
    # total surface area
    num_ba = np.sum([v for v in nrn.iter_segments(mm.segment_area, TreeType.basal_dendrite)])
    num_ap = np.sum([v for v in nrn.iter_segments(mm.segment_area, TreeType.apical_dendrite)])
    results["bb_total_surface_area_apical"] = num_ap
    results["bb_total_surface_area_basal"] = num_ba
    results["bb_total_surface_area_dendrite"] = num_ap + num_ba

    ####################################################################
    # total volume
    num_ba = 0
    num_ap = 0
    basal = [v for v in nrn.iter_segments(mm.segment_volume, TreeType.basal_dendrite)]
    if len(basal) > 0:
        num_ba = np.sum(basal)
    apical = [v for v in nrn.iter_segments(mm.segment_volume, TreeType.apical_dendrite)]
    if len(apical) > 0:
        num_ap = np.sum(apical)
    results["bb_total_volume_apical"] = num_ap
    results["bb_total_volume_basal"] = num_ba
    results["bb_total_volume_dendrite"] = num_ap + num_ba

    ####################################################################
    # total length
    num_ba = np.sum([v for v in nrn.iter_segments(mm.segment_length, TreeType.basal_dendrite)])
    num_ap = np.sum([v for v in nrn.iter_segments(mm.segment_length, TreeType.apical_dendrite)])
    results["bb_total_length_apical"] = num_ap
    results["bb_total_length_basal"] = num_ba
    results["bb_total_length_dendrite"] = num_ap + num_ba

    ####################################################################
    # max radial distance
    num_ba = 0
    num_ap = 0
    basal = [v for v in nrn.get_section_radial_distances(neurite_type=TreeType.basal_dendrite)]
    if len(basal) > 0:
        num_ba = max(basal)
    apical = [v for v in nrn.get_section_radial_distances(neurite_type=TreeType.apical_dendrite)]
    if len(apical) > 0:
        num_ap = max(apical)
    results["bb_max_radial_distance_apical"] = num_ap
    results["bb_max_radial_distance_basal"] = num_ba
    results["bb_max_radial_distance_dendrite"] = num_ap + num_ba

    ####################################################################
    # max path length
    num_ba = 0
    num_ap = 0
    basal = [v for v in nrn.get_section_path_distances(neurite_type=TreeType.basal_dendrite)]
    if len(basal) > 0:
        num_ba = max(basal)
    apical = [v for v in nrn.get_section_path_distances(neurite_type=TreeType.apical_dendrite)]
    if len(apical) > 0:
        num_ap = max(apical)
    results["bb_max_path_length_apical"] = num_ap
    results["bb_max_path_length_basal"] = num_ba
    results["bb_max_path_length_dendrite"] = num_ap + num_ba

    ####################################################################
    # number of branches
    num_ba = 0
    num_ap = 0
    basal = nrn.get_n_sections_per_neurite(TreeType.basal_dendrite)
    if len(basal) > 0:
        num_ba = np.sum(basal)
    apical = nrn.get_n_sections_per_neurite(TreeType.apical_dendrite)
    if len(apical) > 0:
        num_ap = np.sum(apical)
    results["bb_number_branches_apical"] = num_ap
    results["bb_number_branches_basal"] = num_ba
    results["bb_number_branches_dendrite"] = num_ap + num_ba

    ####################################################################
    # Trunk diameter of branches
    num_ap = 0
    num_ba = 0
    basal = nrn.get_trunk_radii(TreeType.apical_dendrite)
    if len(basal) > 0:
        num_ba = 2.0 * np.mean(basal)
    apical = nrn.get_trunk_radii(TreeType.apical_dendrite)
    if len(apical) > 0:
        num_ap = 2.0 * np.mean(apical)
    results["bb_mean_trunk_diameter_apical"] = num_ap
    results["bb_mean_trunk_diameter_basal"] = num_ba
    results["bb_mean_trunk_diameter_dendrite"] = num_ap + num_ba

    ####################################################################
    # Max branch order
    num_ba = 0
    num_ap = 0
    trees = nrn.neurites
    for i in range(len(trees)):
        tree = trees[i]
        partitions = mt.partition(tree)
        if len(partitions) > 0:
            parts = max(partitions)
            tp = tree.value[COLS.TYPE]
            if tp == 3:
                num_ba = max(num_ba, parts)
            elif tp == 4:
                num_ap = max(num_ap, parts)
    results["bb_max_branch_order_apical"] = num_ap
    results["bb_max_branch_order_basal"] = num_ba
    results["bb_max_branch_order_dendrite"] = max(num_ap, num_ba)

    # moments for apical dendrite
    first, second = calculate_moments_on_axis(nrn, TYPE_APICAL, AXIS_X)
    results["bb_first_moment_x_apical"] = first
    results["bb_second_moment_x_apical"] = second
    first, second = calculate_moments_on_axis(nrn, TYPE_APICAL, AXIS_Y)
    results["bb_first_moment_y_apical"] = first
    results["bb_second_moment_y_apical"] = second
    first, second = calculate_moments_on_axis(nrn, TYPE_APICAL, AXIS_Z)
    results["bb_first_moment_z_apical"] = first
    results["bb_second_moment_z_apical"] = second
    # moments for basal dendrite
    first, second = calculate_moments_on_axis(nrn, TYPE_BASAL, AXIS_X)
    results["bb_first_moment_x_basal"] = first
    results["bb_second_moment_x_basal"] = second
    first, second = calculate_moments_on_axis(nrn, TYPE_BASAL, AXIS_Y)
    results["bb_first_moment_y_basal"] = first
    results["bb_second_moment_y_basal"] = second
    first, second = calculate_moments_on_axis(nrn, TYPE_BASAL, AXIS_Z)
    results["bb_first_moment_z_basal"] = first
    results["bb_second_moment_z_basal"] = second
    # moments for entire dendrite
    first, second = calculate_moments_on_axis(nrn, TYPE_DENDRITE, AXIS_X)
    results["bb_first_moment_x_dendrite"] = first
    results["bb_second_moment_x_dendrite"] = second
    first, second = calculate_moments_on_axis(nrn, TYPE_DENDRITE, AXIS_Y)
    results["bb_first_moment_y_dendrite"] = first
    results["bb_second_moment_y_dendrite"] = second
    first, second = calculate_moments_on_axis(nrn, TYPE_DENDRITE, AXIS_Z)
    results["bb_first_moment_z_dendrite"] = first
    results["bb_second_moment_z_dendrite"] = second
    # absolute moments for apical, basal and dendrite (because alignment off)
    first, second = calculate_moments(nrn, TYPE_DENDRITE)
    results["bb_first_moment_dendrite"] = first
    results["bb_second_moment_dendrite"] = second
    first, second = calculate_moments(nrn, TYPE_BASAL)
    results["bb_first_moment_basal"] = first
    results["bb_second_moment_basal"] = second
    first, second = calculate_moments(nrn, TYPE_APICAL)
    results["bb_first_moment_apical"] = first
    results["bb_second_moment_apical"] = second


    #feat = []
    #desc = []
    #for k in results:
    #    desc.append(k)
    #desc.sort()
    #for i in range(len(desc)):
    #    feat.append(results[desc[i]])
    #return feat, desc
    return results

# compute custom features to explore in Tim's embedinator
def compute_embedinator_features(results):
    frst = results["bb_first_moment_x_apical"] 
    sec = results["bb_second_moment_x_apical"] 
    if frst != float('nan') and frst > 0 and sec != float('nan') and sec > 0:
        results["kg_ratio_moment_x_apical"] = sec / frst
    else:
        results["kg_ratio_moment_x_apical"] = float('nan')

    frst = results["bb_first_moment_y_apical"] 
    sec = results["bb_second_moment_y_apical"] 
    if frst != float('nan') and frst > 0 and sec != float('nan') and sec > 0:
        results["kg_ratio_moment_y_apical"] = sec / frst
    else:
        results["kg_ratio_moment_y_apical"] = float('nan')

    frst = results["bb_first_moment_z_apical"] 
    sec = results["bb_second_moment_z_apical"] 
    if frst != float('nan') and frst > 0 and sec != float('nan') and sec > 0:
        results["kg_ratio_moment_z_apical"] = sec / frst
    else:
        results["kg_ratio_moment_z_apical"] = float('nan')

    frst = results["bb_first_moment_apical"] 
    sec = results["bb_second_moment_apical"] 
    if frst != float('nan') and frst > 0 and sec != float('nan') and sec > 0:
        results["kg_ratio_moment_apical"] = sec / frst
    else:
        results["kg_ratio_moment_apical"] = float('nan')



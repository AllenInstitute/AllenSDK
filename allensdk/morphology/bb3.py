from neurom import ezy
import neurom.analysis.morphtree as mt
from neurom.analysis import morphmath as mm
from neurom.core import tree as tr
from neurom.core.dataformat import COLS
from neurom.core.types import TreeType
from neurom.view.view import neuron
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
            # sz is length of segment on specified axis
            sz = abs(seg[0].value[axis] - seg[1].value[axis])
            # pz is centroid of segment on specified axis (midpoint 
            #   between this point and child point
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

def calculate_centroid(nrn, seg_type):
    iterator = nrn.iter_neurites(tr.isegment)
    px = 0.0
    py = 0.0
    pz = 0.0
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
            px += seg[0].value[COL_X]
            py += seg[0].value[COL_Y]
            pz += seg[0].value[COL_Z]
            cnt += 1.0
    if cnt > 0.0:
        px /= cnt
        py /= cnt
        pz /= cnt
    return (px, py, pz)

# for each of 4 dimensions (absolute, x, y, z), calculates the centroid
#   of the specified segment type (eg, basal, apical, axon) and then
#   calculates the mean and standard deviation of branch locations from
#   this point
# return value is the mean, standard deviation and the centroid's distance
#   from the reference location for each dimension
def calculate_branch_center_z(nrn, reference, seg_type):
    # TODO what is difference between morphology fork and bifurcation???
    #iterator = nrn.iter_neurites(tr.ibifurcation_point)
    iterator = nrn.iter_neurites(tr.iforking_point)
    mean = 0.0
    cnt = 0.0
    center = calculate_centroid(nrn, seg_type)
    # calculate stdev
    stdev = 0.0
    for seg in iterator:
        # see if this segment is part of search criteria
        st = seg.value[COL_T]
        if seg_type == TYPE_DENDRITE and (st==TYPE_BASAL or st==TYPE_APICAL):
            match = True
        elif st == seg_type:
            match = True
        else:
            match = False
        # if segment part of query, update moment calculation
        if match:
            dz = center[COL_Z] - seg.value[COL_Z]
            mean += abs(dz)
            cnt += 1.0
    if cnt == 0:
        return [0,0,0]
    mean /= cnt
    iterator = nrn.iter_neurites(tr.iforking_point)
    for seg in iterator:
        # see if this segment is part of search criteria
        st = seg.value[COL_T]
        if seg_type == TYPE_DENDRITE and (st==TYPE_BASAL or st==TYPE_APICAL):
            match = True
        elif st == seg_type:
            match = True
        else:
            match = False
        # if segment part of query, update moment calculation
        if match:
            dist_h = center[COL_Z] + mean - seg.value[COL_Z]
            dist_l = center[COL_Z] - mean - seg.value[COL_Z]
            if abs(dist_h) < abs(dist_l):
                dz = dist_h
            else:
                dz = dist_l
            stdev += dz*dz
    stdev = math.sqrt(stdev / cnt)
    dist = math.sqrt(math.pow((reference[2] - center[2]), 2))
    return [mean, stdev, dist]

## for each of 4 dimensions (absolute, x, y, z), calculates the centroid
##   of the specified segment type (eg, basal, apical, axon) and then
##   calculates the mean and standard deviation of branch locations from
##   this point
## return value is the mean, standard deviation and the centroid's distance
##   from the reference location for each dimension
#def calculate_branch_center(nrn, reference, seg_type):
#    # TODO what is difference between morphology fork and bifurcation???
#    #iterator = nrn.iter_neurites(tr.ibifurcation_point)
#    iterator = nrn.iter_neurites(tr.iforking_point)
#    mean = 0.0
#    meanx = 0.0
#    meany = 0.0
#    meanz = 0.0
#    cnt = 0.0
#    center = calculate_centroid(nrn, seg_type)
#    # calculate stdev
#    stdev = 0.0
#    stdevx = 0.0
#    stdevy = 0.0
#    stdevz = 0.0
#    for seg in iterator:
#        # see if this segment is part of search criteria
#        st = seg.value[COL_T]
#        if seg_type == TYPE_DENDRITE and (st==TYPE_BASAL or st==TYPE_APICAL):
#            match = True
#        elif st == seg_type:
#            match = True
#        else:
#            match = False
#        # if segment part of query, update moment calculation
#        if match:
#            dx = center[COL_X] - seg.value[COL_X]
#            dy = center[COL_Y] - seg.value[COL_Y]
#            dz = center[COL_Z] - seg.value[COL_Z]
#            mean += math.sqrt(dx*dx + dy*dy + dz*dz)
#            meanx += abs(dx)
#            meany += abs(dy)
#            meanz += abs(dz)
#            cnt += 1.0
#    if cnt == 0:
#        return [0,0,0],[0,0,0],[0,0,0],[0,0,0]
#    mean /= cnt
#    meanx /= cnt
#    meany /= cnt
#    meanz /= cnt
#    iterator = nrn.iter_neurites(tr.iforking_point)
#    print("mean")
#    print(mean)
#    print(meanx)
#    print(meany)
#    print(meanz)
#    print("center")
#    print(center[COL_Z])
#    print("mean distance")
#    print(meanz)
#    for seg in iterator:
#        # see if this segment is part of search criteria
#        st = seg.value[COL_T]
#        if seg_type == TYPE_DENDRITE and (st==TYPE_BASAL or st==TYPE_APICAL):
#            match = True
#        elif st == seg_type:
#            match = True
#        else:
#            match = False
#        # if segment part of query, update moment calculation
#        if match:
#            dx = center[COL_X] + meanx - seg.value[COL_X]
#            dy = center[COL_Y] + meany - seg.value[COL_Y]
#            dz = center[COL_Z] + meanz - seg.value[COL_Z]
#            stdev += dx*dx + dy*dy + dz*dz
#            stdevx += dx*dx
#            stdevy += dy*dy
#            stdevz += dz*dz
#            print("%f\t%f" % (seg.value[COL_Z], dz))
#    stdev = math.sqrt(stdev / cnt)
#    stdevx = math.sqrt(stdevx / cnt)
#    stdevy = math.sqrt(stdevy / cnt)
#    stdevz = math.sqrt(stdevz / cnt)
#    dist = np.linalg.norm(np.subtract(reference, center))
#    distx = math.sqrt(math.pow((reference[0] - center[0]), 2))
#    disty = math.sqrt(math.pow((reference[1] - center[1]), 2))
#    distz = math.sqrt(math.pow((reference[2] - center[2]), 2))
#    res = [mean, stdev, dist]
#    resx = [meanx, stdevx, distx]
#    resy = [meany, stdevy, disty]
#    resz = [meanz, stdevz, distz]
#    return res, resx, resy, resz

def compute_features(swc_file, spec_id=None):
    nrn = ezy.load_neuron(swc_file)
    if spec_id is not None:
        fig, ax = neuron(nrn, plane='xz')
        outfile = "images/" + str(spec_id) + "_" + swc_file.split('/')[-1] + ".png"
        print("Saving '%s'" % outfile)
        fig.savefig(outfile)
    results = {}

    ####################################################################
    center = calculate_branch_center_z(nrn, nrn.soma.center, TYPE_APICAL)
    results["kg_branch_mean_from_centroid_z_apical"] = center[0]
    results["kg_branch_stdev_from_centroid_z_apical"] = center[1]
    results["kg_branch_centroid_distance_z_apical"] = center[2]
    results["kg_soma_depth"] = nrn.soma.center[2]

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
    basal = nrn.get_trunk_radii(TreeType.basal_dendrite)
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

    return results

# compute custom features to explore in Tim's embedinator
def compute_embedinator_features(results):
    mom = results["bb_first_moment_z_apical"] 
    dist = results["bb_max_radial_distance_apical"] 
    bran = results["bb_number_branches_apical"]
    if mom != float('nan') and mom > 0 and dist != float('nan'):
        results["kg_radial_dist_over_moment_z_apical"] = dist / mom
    else:
        results["kg_radial_dist_over_moment_z_apical"] = 0

    if dist != float('nan') and bran != float('nan') and bran > 0:
        results["kg_num_branches_over_radial_dist_apical"] = dist / bran
        results["kg_centroid_over_radial_dist_apical"] = results["kg_branch_mean_from_centroid_z_apical"] / results["bb_max_radial_distance_apical"]
        results["kg_mean_over_centroid"] = results["kg_branch_mean_from_centroid_z_apical"] / results["kg_branch_centroid_distance_z_apical"]
        results["kg_mean_over_stdev"] = results["kg_branch_mean_from_centroid_z_apical"] / results["kg_branch_stdev_from_centroid_z_apical"]
    else:
        results["kg_num_branches_over_radial_dist_apical"] = 0
        results["kg_centroid_over_radial_dist_apical"] = 0
        results["kg_mean_over_centroid"] = 0
        results["kg_mean_over_stdev"] = 0




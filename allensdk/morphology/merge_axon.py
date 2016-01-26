#!/usr/bin/python
import glob
import traceback
from subprocess import call
import sys
from swc import *

########################################################################
#
# script to merge 2 swc files
# It merges the dendrite morphology from the first file with the axon
#   morphology of the second. All axons nodes are removed from the 
#   dendrite morphology and that is replaced by the 2nd axon morphology
#

DEND_SUFFIX = "Dendrite.swc"
AXON_SUFFIX = "DendriteAxon.swc"
MERGE_SUFFIX = "Merged.swc"

def merge_files(master_dend, master_axon, combined):
    print("Reading master dendrite file '%s'" % master_dend)
    final_dend = read_swc(master_dend)
    print("Reading master axon file '%s'" % master_axon)
    final_axon = read_swc(master_axon)

    # copy dendrites from master dendrite into ouptut morphology object
    merged = Morphology(final_dend.compartment_list)

    # strip axons from output file -- these will be replaced with axons
    #   from the master axon file
    merged.strip_type(2)

    # strip everything except axons from the master axon file
    final_axon.strip_all_other_types(2, keep_soma=False)

    # add master axons to the master dendrites
    merged.append(final_axon.compartment_list)

    # reconnect axon and dendrite nodes
    # use final dendrite file as connectivity template
    merge_types = []
    for seg in final_dend.compartment_list:
        merge = False
        # look for axon roots in the original file
        # when finding one, look for the equivalent child and parent nodes in 
        #   the split files. Then merge the axon node with a parent 
        #   soma/dendrite node, or a dendrite node with a parent axon
        #
        # look for axon that has a parent dendrite or soma node
        merge_type = ""
        if seg["type"] == 2:
            par = final_dend.parent_of(seg)
            if par is not None:
                if par["type"] == 1:
                    merge_type = "axon to soma"
                    merge = True
                elif par["type"] == 3:
                    merge_type = "axon to dendrite"
                    merge = True
        # look for a dendrite that has an axon parent
        elif seg["type"] == 3:
            par = final_dend.parent_of(seg)
            if par is not None and par["type"] == 2:
                merge_type = "dendrite to axon"
                merge = True
        # if merge specified, do it
        if merge:
            old_par = merged.find(par[NODE_X], par[NODE_Y], par[NODE_Z], 2.0)
            if len(old_par) != 1:
                print("** Position mismatch of parent node between files")
                print("   Found %d possible matches" % len(old_par))
                print("")
                print("Master dendrite file nodes:")
                print_node(seg)
                print_node(par)
                print("")
                print("Master axon file matches:")
                for x in old_par:
                    print_node(x)
                continue
                print("")
            old_child = merged.find(seg[NODE_X], seg[NODE_Y], seg[NODE_Z], 2.0)
            if len(old_child) != 1:
                print("** Position mismatch of root axon node between files")
                print("   Found %d possible matches" % len(old_child))
                print("")
                print("Master dendrite file nodes:")
                print_node(seg)
                print_node(par)
                print("")
                print("Master axon file matches:")
                for x in old_child:
                    print_node(x)
                continue
            merge_types.append(merge_type)
            merged.change_parent(old_child[0], old_par[0])
    plural = "s"
    if len(merge_types) == 1:
        plural = ""
    print("Merged %d root%s" % (len(merge_types), plural))
    for s in merge_types:
        print("\t%s" % s)
    print("Writing '%s'" % combined)
    merged.write(combined)

def usage():
    print("Usage: python %s [file name]" % sys.argv[0])
    print("")
    print("File name can be one of master dendrite or axon files")
    print("    of the form *_p_X.swc, where X is one of '%s'" % DEND_SUFFIX)
    print("    or '%s'" % AXON_SUFFIX)
    print("If no file name is provided, all files of the pattern ")
    print("    *_p_%s in the current working directory" % DEND_SUFFIX)
    print("    will be processed")
    sys.exit(1)

def unit_test_1():
    swc = ""
    swc += "0 1 0.0 0.0 0.0 10.0 -1\n"
    swc += "1 2 10.0 0.1 0.1 1.0 0\n"
    swc += "2 2 20.0 0.5 0.5 1.0 1\n"
    swc += "3 3 0.4 10.5 0.2 1.0 0\n"
    swc += "4 3 0.1 20.5 0.1 1.0 3\n"
    dfile = "test1_dend.swc"
    open(dfile, "w").write(swc)
    swc += "5 2 30.0 0.1 0.2 1.0 2\n"
    afile = "test1_axon.swc"
    open(afile, "w").write(swc)
    mfile = "test1_merge.swc"
    merge_files(dfile, afile, mfile)
    call(["cat", mfile])
    print("-> verify that axon at 30,0,0 linked to axon at 20,0,0")
    print("-> verify that axon at 10,0,0 linked to soma at 0,0,0")
    print("")

def unit_test_2():
    swc = ""
    swc += "0 1 0.0 0.0 0.0 10.0 -1\n"
    swc += "1 2 10.0 0.1 0.1 1.0 0\n"
    swc += "2 2 20.0 0.5 0.5 1.0 1\n"
    swc += "3 3 0.4 10.5 0.2 1.0 1\n"
    swc += "4 3 0.1 20.5 0.1 1.0 3\n"
    dfile = "test2_dend.swc"
    open(dfile, "w").write(swc)
    swc += "5 2 30.0 0.1 0.2 1.0 2\n"
    afile = "test2_axon.swc"
    open(afile, "w").write(swc)
    mfile = "test2_merge.swc"
    merge_files(dfile, afile, mfile)
    call(["cat", mfile])
    print("-> verify that dendrite at 0,10,0 linked to axon at 10,0,0")
    print("")

def unit_test_3():
    swc = ""
    swc += "0 1 0.0 0.0 0.0 10.0 -1\n"
    swc += "1 2 10.0 0.1 0.1 1.0 3\n"
    swc += "2 2 20.0 0.5 0.5 1.0 1\n"
    swc += "3 3 0.4 10.5 0.2 1.0 0\n"
    swc += "4 3 0.1 20.5 0.1 1.0 3\n"
    dfile = "test3_dend.swc"
    open(dfile, "w").write(swc)
    swc += "5 2 30.0 0.1 0.2 1.0 2\n"
    afile = "test3_axon.swc"
    open(afile, "w").write(swc)
    mfile = "test3_merge.swc"
    merge_files(dfile, afile, mfile)
    call(["cat", mfile])
    print("-> verify that axon at 10,0,0 linked to dendrite at 0,10,0")
    print("")

def unit_tests():
    print("###########################################")
    print("Unit-test mode")
    print("###########################################")
    unit_test_1()
    print("###########################################")
    unit_test_2()
    print("###########################################")
    unit_test_3()
    print("###########################################")

axon_files = []
dend_files = []
combined_files = []

if len(sys.argv) == 2:
    if sys.argv[1] == "TEST":
        unit_tests()
    else:
        axon_files.append(sys.argv[1] + AXON_SUFFIX)
        dend_files.append(sys.argv[1] + DEND_SUFFIX)
        combined_files.append(sys.argv[1] + MERGE_SUFFIX)
elif len(sys.argv) == 1:
    name_expr = "*_p_%s" % DEND_SUFFIX
    #print("Looking for files like '%s'" % name_expr)
    files = glob.glob(name_expr)
    for f in files:
        root = f.split('_p_')[0] + "_p_"
        axon_files.append(root + AXON_SUFFIX)
        dend_files.append(root + DEND_SUFFIX)
        combined_files.append(root + MERGE_SUFFIX)
else:
    usage()

for i in range(len(dend_files)):
    d = dend_files[i]
    a = axon_files[i]
    m = combined_files[i]
    try:
        merge_files(d, a, m)
        print("")
    except IOError:
        print("** Error creating merge file")
        traceback.print_exc()


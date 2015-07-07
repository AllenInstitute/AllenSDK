#!/usr/bin/python
import h5py
import sys

# load attributes
def read_attributes(hval):
    attr = {}
    for k in hval.attrs:
        attr[k] = type(hval.attrs[k])
    return attr

# returns summary of group. 
# the only element for comparison here is the group's attributes
def read_group(hval):
    desc = {}
    desc["attr"] = read_attributes(hval)
    desc["htype"] = "group"
    return desc

# returns summary of dataset
# the only elements for comparison here are the dataset's attributes,
#   and the dataset dtype
def read_data(hval):
    desc = {}
    desc["attr"] = read_attributes(hval)
    desc["htype"] = "dataset"
    desc["dtype"] = type(hval.value)
    return desc

# creates and returns a summary description for every element in a group
def evaluate_group(path, grp):
    desc = {}
    for k,v in grp.iteritems():
        if isinstance(v, h5py.Dataset):
            desc[k] = read_data(v)
        elif isinstance(v, h5py.Group):
            desc[k] = read_group(v)
        else:
            fatal_error("Unknown h5py type: %s (%s -- %s)"% (type(v), path, k))
    return desc

def diff_groups(file1, grp1, file2, grp2, path):
    print "------------------------------"
    print "Examining " + path
    desc1 = evaluate_group(path, grp1)
    desc2 = evaluate_group(path, grp2)
    common = []
    for k in desc1:
        if k in desc2:
            common.append(k)
        else:
            print "** Element '%s' only in '%s' (DIFF_UNIQUE_A)**" % (k, file1)
    for k in desc2:
        if k not in desc1:
            print "** Element '%s' only in '%s' (DIFF_UNIQUE_B)**" % (k, file2)
    for i in range(len(common)):
        name = common[i]
        print "\t" + name
        # compare types
        h1 = desc1[name]["htype"]
        h2 = desc2[name]["htype"]
        if h1 != h2:
            print "**  Different element types: '%s' and '%s' (DIFF_OBJECTS)" % (h1, h2)
            continue    # different hdf5 types -- don't try to compare further
        if h1 != "dataset" and h1 != "group":
            print "WARNING: element is not a recognized type (%s) and isn't being evaluated" % h1
            continue
        # handle datasets first
        if desc1[name]["htype"] != "dataset":
            continue
        # compare data
        fld1 = desc1[name]
        if desc1[name]["dtype"] != desc2[name]["dtype"]:
            d1 = desc1[name]["dtype"]
            d2 = desc2[name]["dtype"]
            print "** Different dtypes: '%s' and '%s' (DIFF_DTYPE)**" % (d1, d2)
        # compare attributes
        for k in desc1[name]["attr"]:
            if k not in desc2[name]["attr"]:
                print "** Attribute '%s' only in '%s' (DIFF_UNIQ_ATTR_A)**" % (k, file1)
        for k in desc2[name]["attr"]:
            if k not in desc1[name]["attr"]:
                print "** Attribute '%s' only in '%s' (DIFF_UNIQ_ATTR_B)**" % (k, file2)
        for k in desc1[name]["attr"]:
            if k in desc2[name]["attr"]:
                v = desc1[name]["attr"][k]
                v2 = desc2[name]["attr"][k]
                if v != v2:
                    print "** Attribute '%s' has different type: '%s' and '%s' (DIFF_ATTR_DTYPE)" % (k, v, v2)
    for i in range(len(common)):
        name = common[i]
        # compare types
        if desc1[name]["htype"] != desc2[name]["htype"]:
            continue    # problem already reported
        if desc1[name]["htype"] != "group":
            continue
        # compare attributes
        for k in desc1[name]["attr"]:
            if k not in desc2[name]["attr"]:
                print "** Attribute '%s' only in '%s' (DIFF_UNIQ_ATTR_A)**" % (k, file1)
        for k in desc2[name]["attr"]:
            if k not in desc1[name]["attr"]:
                print "** Attribute '%s' only in '%s' (DIFF_UNIQ_ATTR_B)**" % (k, file2)
        # recurse into subgroup
        diff_groups(file1, grp1[name], file2, grp2[name], path+name+"/")


def diff_files(file1, file2):
    print "Comparing '%s' and '%s'" % (file1, file2)
    try:
        f1 = h5py.File(file1, 'r')
    except IOError:
        print "Unable to open file '%s'" % file1
        sys.exit(1)
    try:
        f2 = h5py.File(file2, 'r')
    except IOError:
        print "Unable to open file '%s'" % file2
        sys.exit(1)
    diff_groups(file1, f1["/"], file2, f2["/"], "/")

if len(sys.argv) != 3:
    print "Usage: %s <file1.h5> <file2.h5>" % sys.argv[0]
    sys.exit(1)

diff_files(sys.argv[1], sys.argv[2])



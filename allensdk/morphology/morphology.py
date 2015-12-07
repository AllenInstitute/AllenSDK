#!/usr/bin/python
import morphology_analysis as morphology
import json
import sys

def usage():
    print("This script generates feature data from neuron morphology data")
    if sys.argv[0].startswith("./"):
        name = sys.argv[0][2:]
    else:
        name = sys.argv[0]
    print("Usage: %s <input.json> <output.json>" % name)
    sys.exit(1)

# minimum input to script is script name, output file name and one swc file
# if we don't have at least this, abort
if len(sys.argv) != 3:
    usage()

infile = sys.argv[1]
try:
    jin = json.load(open(infile, "r"))
except:
    print("Unable to open/read file '%s' as json" % infile)
    sys.exit(1)

# find swc file in jin[well_known_files]
wkf = jin["well_known_files"]
swc_file = None
for i in range(len(wkf)):
    if "filename" in wkf[i] and wkf[i]["filename"].endswith(".swc"):
        if swc_file is not None:
            print("Error -- multiple .swc files specified in %s" % infile)
            print("Don't know which one to process")
            sys.exit(1)
        swc_file = wkf[i]["storage_directory"] + wkf[i]["filename"]


# dictionary of feature data
morph_data = {}

# description of what features are being extracted
# presently there are two categories -- physical features and geometric moments
feature_desc = None
gmi_desc = None

# load segment data into memory
print("Processing '%s'" % swc_file)
nrn = morphology.SWC(swc_file)

# apply pia transform
# algorithm inferred from v3d:apply_transform_func.cpp:apply_transform()
#   and Xiaoxiao's script to run v3d
# foreach segment:
#   x = x*tvr_00 + y*tvr_01 + z*tvr_02 + tvr_09
#   y = x*tvr_03 + y*tvr_04 + z*tvr_05 + tvr_10
#   z = x*tvr_06 + y*tvr_07 + z*tvr_08 + tvr_11
xf = jin["pia_transform"]
for i in range(len(nrn.obj_list)):
    seg = nrn.obj_list[i]
    x = seg.x
    y = seg.y
    z = seg.z
    seg.x = x*xf["tvr_00"] + y*xf["tvr_01"] + z*xf["tvr_02"] + xf["tvr_09"]
    seg.y = x*xf["tvr_03"] + y*xf["tvr_04"] + z*xf["tvr_05"] + xf["tvr_10"]
    seg.z = x*xf["tvr_06"] + y*xf["tvr_07"] + z*xf["tvr_08"] + xf["tvr_11"]

# calculate features
gmi, gmi_desc = morphology.computeGMI(nrn)
gmi_out = {}
for i in range(len(gmi)):
    gmi_out[gmi_desc[i]] = gmi[i]
morph_data["gmi"] = gmi_out
#
features, feature_desc = morphology.computeFeature(nrn)
feat_out = {}
for i in range(len(features)):
    feat_out[feature_desc[i]] =  features[i]
morph_data["features"] = feat_out

# make sure nothing bad happened in analysis
if feature_desc is None or gmi_desc is None:
    print("Internal error -- bailing out")
    sys.exit(1)

jout = jin
if "morphology_data" in jout:
    del jout["morphology_data"]
jout["morphology_data"] = morph_data
try:
    with open(sys.argv[2], "w") as f:
        json.dump(jout, f, indent=2)
        f.close()
except:
    print("Error writing output json file %s" % sys.argv[2])
    sys.exit(1)



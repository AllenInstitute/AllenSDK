#!/usr/bin/python
import morphology_analysis as morphology
import traceback
import json
import sys

def usage():
    print("This script generates feature data from a neuron SWC morphology")
    if sys.argv[0].startswith("./"):
        name = sys.argv[0][2:]
    else:
        name = sys.argv[0]
    print("Usage: %s <input.json> <output.json>" % name)
    sys.exit(1)

if len(sys.argv) != 3:
    usage()

infile = sys.argv[1]
outfile = sys.argv[2]
try:
    jin = json.load(open(infile, "r"))
except:
    print("Unable to open/read file '%s' as json" % infile)
    sys.exit(1)

# get the input file
swc_file = jin["swc_file"]

# dictionary of feature data
morph_data = {}

# description of what features are being extracted
# presently there are two categories -- physical features and geometric moments
feature_desc = None
gmi_desc = None

# load segment data into memory
try:
    print("Processing '%s'" % swc_file)
    nrn = morphology.SWC(swc_file)
except:
    print("*************************")
    print("Error loading SWC file into memory")
    raise

# apply transform to 'pia-space'
try:
    xf = jin["pia_transform"]
    affine = []
    affine.append(1000.0 * xf["tvr_00"])
    affine.append(1000.0 * xf["tvr_01"])
    affine.append(1000.0 * xf["tvr_02"])
    affine.append(1000.0 * xf["tvr_03"])
    affine.append(1000.0 * xf["tvr_04"])
    affine.append(1000.0 * xf["tvr_05"])
    affine.append(1000.0 * xf["tvr_06"])
    affine.append(1000.0 * xf["tvr_07"])
    affine.append(1000.0 * xf["tvr_08"])
    affine.append(1000.0 * xf["tvr_09"])
    affine.append(1000.0 * xf["tvr_10"])
    affine.append(1000.0 * xf["tvr_11"])
except:
    print("*************************")
    print("Error reading affine transform from input json file")
    traceback.print_exc()
    sys.exit(1)
try:
    nrn.apply_affine(affine)
except:
    print("*************************")
    print("Error reading affine transform from input json file")
    traceback.print_exc()
    sys.exit(1)
#nrn.save_to("bar.swc")

########################################################################
#
# calculate features
# GMI (moments)
try:
    gmi, gmi_desc = morphology.computeGMI(nrn)
    gmi_out = {}
    for i in range(len(gmi)):
        gmi_out[gmi_desc[i]] = gmi[i]
    morph_data["gmi"] = gmi_out
except:
    print("*************************")
    print("Error calculating GMI features")
    traceback.print_exc()
    sys.exit(1)
# morphology features
try:
    features, feature_desc = morphology.computeFeature(nrn)
    feat_out = {}
    for i in range(len(features)):
        feat_out[feature_desc[i]] =  features[i]
    morph_data["features"] = feat_out
except:
    print("*************************")
    print("Error calculating GMI features")
    traceback.print_exc()
    sys.exit(1)

# make sure nothing bad happened in analysis
if feature_desc is None or gmi_desc is None:
    print("*************************")
    print("Internal error -- bailing out")
    sys.exit(1)

########################################################################
# output the data
jout = jin
if "morphology_data" in jout:
    del jout["morphology_data"]
jout["morphology_data"] = morph_data
try:
    with open(outfile, "w") as f:
        json.dump(jout, f, indent=2)
        f.close()
except:
    print("Error writing output json file %s" % outfile)
    sys.exit(1)



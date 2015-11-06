#!/usr/bin/python
import morphology_analysis as morphology
import json
import sys

def usage():
    print("This script generates feature data from neuron morphology data")
    print("Input to the script is one or more .swc files plus the name of")
    print("    the desired output file. Presently, the output formats of")
    print("    .csv and .json are supported")
    print("")
    if sys.argv[0].startswith("./"):
        name = sys.argv[0][2:]
    else:
        name = sys.argv[0]
    print("Usage: %s <output file> <file1.swc> [file2.swc [file3.swc [...]]]" % name)
    sys.exit(1)

# minimum input to script is script name, output file name and one swc file
# if we don't have at least this, abort
if len(sys.argv) < 3:
    usage()

# make sure desired output file is of supported type
if not sys.argv[1].endswith('json') and not sys.argv[1].endswith('csv'):
    usage()

# array storing a dictionary of feature data. one array entry per file
morph_data = []

# description of what features are being extracted
# presently there are two categories -- physical features and geometric moments
feature_desc = None
gmi_desc = None

# calculate features and load into memory
for i in range(2,len(sys.argv)):
    print("Processing '%s'" % sys.argv[i])
    nrn_data = {}
    nrn_data["filename"] = sys.argv[i]
    #
    nrn = morphology.SWC(sys.argv[i])
    gmi, gmi_desc = morphology.computeGMI(nrn)
    gmi_out = {}
    for i in range(len(gmi)):
        gmi_out[gmi_desc[i]] = gmi[i]
    nrn_data["gmi"] = gmi_out
    #
    features, feature_desc = morphology.computeFeature(nrn)
    feat_out = {}
    for i in range(len(features)):
        feat_out[feature_desc[i]] =  features[i]
    nrn_data["features"] = feat_out
    #
    morph_data.append(nrn_data)

# feature_desc and gmi_desc is defined from the previous analysis
#   loops. those values are used below
# make sure nothing bad happened in that analysis
if feature_desc is None or gmi_desc is None:
    print("Internal error -- bailing out")
    sys.exit(1)

# prepare output for specified format
if sys.argv[1].endswith('csv'):
    try:
        f = open(sys.argv[1], "w")
    except IOError:
        print("Unable to open input file '%s'" % sys.argv[1])
        sys.exit(1)
    # write CSV header row
    f.write("filename,")
    for j in range(len(feature_desc)):
        f.write(feature_desc[j])
        f.write(",")
    for j in range(len(gmi_desc)-1):
        f.write(gmi_desc[j])
        f.write(",")
    f.write(gmi_desc[-1] + "\n")
    # write one row for each file
    for i in range(len(morph_data)):
        # reload feature and gmi data
        # the existing feature_desc and gmi_desc still applies to it
        gmi = morph_data[i]["gmi"]
        features = morph_data[i]["features"]
        f.write(morph_data[i]["filename"] + ",")
        for j in range(len(feature_desc)):
            f.write(str(features[feature_desc[j]]))
            f.write(",")
        for j in range(len(gmi_desc)):
            f.write(str(gmi[gmi_desc[j]]))
            f.write(",")
        f.write("\n")
    f.close()
elif sys.argv[1].endswith('json'):
    output = {}
    output["morphology_data"] = morph_data
    with open(sys.argv[1], "w") as f:
        json.dump(output, f, indent=2)
        f.close()


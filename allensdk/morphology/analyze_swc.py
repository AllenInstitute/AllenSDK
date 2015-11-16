#!/usr/bin/python
import morphology_analysis as morphology
from morphology_analysis_bb import compute_features as compute_features_bb
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

DEBUG = False
if DEBUG:
    outfile = "out.csv"
    swc_files = []
    swc_files.append('/home/keithg/data/preprocessed/Htr3a-Cre_NO152_Ai14_IVSCC_-192819.04.02.01_485995691_m.swc')
else:
    outfile = sys.argv[1]
    swc_files = sys.argv[2:]

# minimum input to script is script name, output file name and one swc file
# if we don't have at least this, abort
if not DEBUG and len(sys.argv) < 3:
    usage()

# make sure desired output file is of supported type
if not outfile.endswith('json') and not outfile.endswith('csv'):
    usage()

# array storing a dictionary of feature data. one array entry per file
morph_data = []

# description of what features are being extracted
# presently there are two categories -- physical features and geometric moments
master_list = None

# calculate features and load into memory
for i in range(len(swc_files)):
    swc_file = swc_files[i]
    print("Processing '%s'" % swc_file)
    try:
        nrn_data = {}
        nrn_data["filename"] = swc_file

        try:
            nrn = morphology.SWC(swc_file)
            gmi, gmi_desc = morphology.computeGMI(nrn)
            gmi_out = {}
            for j in range(len(gmi)):
                gmi_out[gmi_desc[j]] = gmi[j]
            nrn_data["gmi"] = gmi_out
        except:
            print("Error calculating GMI")
            raise
        try:
            features, feature_desc = morphology.computeFeature(nrn)
            feat_out = {}
            for j in range(len(features)):
                feat_out[feature_desc[j]] =  features[j]
            nrn_data["features"] = feat_out
        except:
            print("Error calculating l-measure")
            raise
        #
        try:
            bb_feat, bb_desc = compute_features_bb(swc_file)
            feat_out = {}
            for j in range(len(bb_feat)):
                feat_out[bb_desc[j]] =  bb_feat[j]
            nrn_data["features_bb"] = feat_out
        except:
            print("Error calculating BB features")
            raise
        #
        if master_list is None:
            master_list = []
            for i in range(len(feature_desc)):
                master_list.append(feature_desc[i])
            for i in range(len(gmi_desc)):
                master_list.append(gmi_desc[i])
            for i in range(len(bb_desc)):
                master_list.append(bb_desc[i])
        morph_data.append(nrn_data)
    except Exception as e:
        print("Error processing %s" % swc_file)
        print("Unexpected error:", sys.exc_info()[0])
        if DEBUG:
            raise
        continue

if master_list is None:
    print("Internal error -- bailing out")
    sys.exit(1)

# prepare output for specified format
if outfile.endswith('csv'):
    try:
        f = open(outfile, "w")
    except IOError:
        print("Unable to open input file '%s'" % outfile)
        sys.exit(1)
    # write CSV header row
    f.write("filename,")
    for j in range(len(master_list)-1):
        f.write(master_list[j])
        f.write(",")
    f.write(master_list[-1] + "\n")
    # write one row for each file
    for i in range(len(morph_data)):
        # reload feature and gmi data
        # the existing feature_desc and gmi_desc still applies to it
        gmi = morph_data[i]["gmi"]
        features = morph_data[i]["features"]
        bb_feat = morph_data[i]["features_bb"]
        f.write(morph_data[i]["filename"] + ",")
        for j in range(len(feature_desc)):
            f.write(str(features[feature_desc[j]]))
            f.write(",")
        for j in range(len(gmi_desc)):
            f.write(str(gmi[gmi_desc[j]]))
            f.write(",")
        for j in range(len(bb_desc)):
            f.write(str(bb_feat[bb_desc[j]]))
            f.write(",")
        f.write("\n")
    f.close()
elif outfile.endswith('json'):
    output = {}
    output["morphology_data"] = morph_data
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
        f.close()


#!/usr/bin/python
import morphology_analysis as morph
import json
import sys

if len(sys.argv) == 3:
    jin = sys.argv[1]
    jout = sys.argv[2]
elif len(sys.argv) == 4:
    jin = sys.argv[2]
    jout = sys.argv[3]
else:
    print("Usage: python %s <infile.json> <outfile.json>" % sys.argv[0])
    sys.exit(1)

with open(jin, "r") as f:
    js = json.load(f)
    if "file_name" not in js or "file_path" not in js:
        print("Input json file must specify 'file_name' and 'file_path'")
        sys.exit(1)
    path = js["file_path"]
    if path.endswith('/'):
        fname = path + js["file_name"]
    else:
        fname = path + '/' + js["file_name"]
    f.close()

print("Processing %s" % fname)

nrn = morph.SWC(fname)

output = {}

gmi, gmi_desc = morph.computeGMI(nrn)
gmi_out = {}
for i in range(len(gmi)):
    gmi_out[gmi_desc[i]] = gmi[i]

features, feature_desc = morph.computeFeature(nrn)
feat_out = {}
for i in range(len(features)):
    feat_out[feature_desc[i]] =  features[i]

output["gmi"] = gmi_out
output["features"] = feat_out
output["file_name"] = fname
output["file_path"] = path

with open(jout, "w") as f:
    json.dump(output, f, indent=2)
    f.close()


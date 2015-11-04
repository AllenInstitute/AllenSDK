#!/usr/bin/python
import morphology_analysis as morph
import sys

if len(sys.argv) == 1:
    print("Usage: python %s <file1.swc> [fileN.swc] > out.csv" % sys.argv[0])
    sys.exit(1)

for i in range(1,len(sys.argv)):
    nrn = morph.SWC(sys.argv[i])

    gmi, gmi_desc = morph.computeGMI(nrn)
    features, feature_desc = morph.computeFeature(nrn)

    sys.stdout.write("filename,")
    for j in range(len(feature_desc)):
        sys.stdout.write(feature_desc[j])
        sys.stdout.write(",")
    for j in range(len(gmi_desc)-1):
        sys.stdout.write(gmi_desc[j])
        sys.stdout.write(",")
    print(gmi_desc[-1])

    sys.stdout.write(sys.argv[i] + ",")
    for j in range(len(features)):
        sys.stdout.write(str(features[j]))
        sys.stdout.write(",")
    for j in range(len(gmi)-1):
        sys.stdout.write(str(gmi[j]))
        sys.stdout.write(",")
    print(gmi[-1])



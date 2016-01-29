#!/usr/bin/python
import glob
import traceback
from subprocess import call
import sys
from swc import *

SOURCE_DIR = "/mnt/For_Annotation/AutoTraceFiles/Production/KeithandWayne/Dendrites_MergewithAxons/"
TARGET_DIR = "/mnt/For_Annotation/AutoTraceFiles/Production/KeithandWayne/Dendrites_NeedAxonsRemoved/Stripped_Dendrites_without_Axons/"
#TARGET_DIR = "./Stripped_Dendrites_without_Axons/"

if len(sys.argv) == 1:
    files = glob.glob(SOURCE_DIR + "*_p_*.swc")
else:
    files = []
    for i in range(1, len(sys.argv)):
        files.append(sys.argv[i])

for f in files:
    fname = f.split('/')[-1]
    print("Processing '%s'" % fname)
    try:
        morph = read_swc(f)
        morph.stumpify_axon(count=10)
    except:
        print("** Failed to process file **\n")
        print("** Full path: %s" % f)
        traceback.print_exc()
        continue
    try:
        f_root = fname.split('_p_')[0]
        outfile = TARGET_DIR + f_root + "_p_Stump.swc"
        morph.write(outfile)
    except:
        print("** failed to write file '%s'" + outfile)
        print("** Full path: %s" % f)


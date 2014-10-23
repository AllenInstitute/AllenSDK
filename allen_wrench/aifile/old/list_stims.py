#!/usr/bin/python
import h5py
import sys
from aif_common import list_templates

TESTING = True

if len(sys.argv) != 2:
	if TESTING:
		infile = "157436.03.01.ai"
	else:
		print "Usage: %s <input ai>" % sys.argv[0]
		sys.exit(1)
else:
	infile = sys.argv[1]

f = h5py.File(infile, "r")
stims, lst = list_templates(f)

print "%s\t%28s\t%s\t%s" % ("Number", "Stimulus", "pA", "Sweeps")
for i in range(len(lst)):
	peak = 1.0e12 * stims[lst[i]]["max_value"].value
	swps = stims[lst[i]].attrs["epochs"]
	epochs = ""
	for j in range(len(swps)):
		if j >= 1:
			epochs += ", "
		epochs += swps[j].split('_')[1]
	print "%d\t%28s\t%g\t%s" % (i, lst[i], peak,epochs)

f.close()


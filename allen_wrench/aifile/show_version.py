#!/usr/bin/python
import h5py
import sys

TESTING = True

if len(sys.argv) != 2:
	if TESTING:
		infile = "157436.03.01.ai"
	else:
		print "Usage: %s <ai file>" % sys.argv[0]
		sys.exit(1)
else:
	infile = sys.argv[1]
	image_num = sys.argv[2]

f = h5py.File(infile, "r")
print f["ai_version"].value
f.close()



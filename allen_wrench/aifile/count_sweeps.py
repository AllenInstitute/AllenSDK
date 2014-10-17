#!/usr/bin/python
import h5py
import sys

TESTING = True

if len(sys.argv) != 2:
	if TESTING:
		infile = "foo.h5"
	else:
		print "Usage: %s <input nwb>" % sys.argv[0]
		sys.exit(1)
else:
	infile = sys.argv[1]

f = h5py.File(infile, "r")
epochs = f["epochs"]

last_n = 0
for k in epochs.keys():
	n = k.split('_')[1]
	if n > last_n:
		last_n = n

print last_n

f.close()


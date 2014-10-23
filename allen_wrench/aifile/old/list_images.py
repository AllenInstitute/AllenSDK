#!/usr/bin/python
import h5py
import sys

TESTING = True

if len(sys.argv) != 2:
	if TESTING:
		infile = "157436.03.01.ai2"
	else:
		print "Usage: %s <input ai>" % sys.argv[0]
		sys.exit(1)
else:
	infile = sys.argv[1]

f = h5py.File(infile, "r")
imgs = f["acquisition"]["images"]

lst = []
for k in imgs.keys():
	lst.append(k)
lst.sort()

print "%s\t%40s" % ("Number", "Image")
for i in range(len(lst)):
	print "%d\t%40s" % (i, lst[i])

f.close()


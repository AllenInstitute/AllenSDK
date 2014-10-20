#!/usr/bin/python
import h5py
import sys
from subprocess import call

TESTING = True

image_num = 0

if len(sys.argv) != 3:
	if TESTING:
		infile = "157436.03.01.ai"
		image_num = 3
	else:
		print "Usage: %s <input ai> <image #>" % sys.argv[0]
		sys.exit(1)
else:
	infile = sys.argv[1]
	image_num = sys.argv[2]

f = h5py.File(infile, "r")
imgs = f["acquisition"]["images"]

lst = []
for k in imgs.keys():
	lst.append(k)
lst.sort()

stream = imgs[lst[image_num]].value
fname = lst[image_num]
with open(fname, "wb") as tif:
	tif.write(stream)
	tif.close()
f.close()

call(["eog", fname])



#!/usr/bin/python
import glob
import os
import h5d_convert
import h5py
import sys

h5files = glob.glob('/local2/ephys/*/*h5')

for i in range(len(h5files)):
	infile = h5files[i]
	dirlen = len(os.path.dirname(infile))
	outfile = ""
	if dirlen > 1:
		outfile = infile[dirlen+1:]
	else:
		outfile = infile
	if outfile.endswith(".h5"):
		outfile = outfile[:-3]
	outfile = "/local2/ephys/" + outfile + ".aibs"

	if os.path.isfile(outfile):
		# TODO add version checking here and regenerate file if new version
		continue
	h5d_convert.convert_h5d_aif(infile, outfile)


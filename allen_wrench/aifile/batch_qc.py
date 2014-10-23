#!/usr/bin/python
import glob
from subprocess import call

aifiles = glob.glob('/local2/ephys/*.ai')

for i in range(len(aifiles)):
	call(["./ephys_qc.py", aifiles[i]])


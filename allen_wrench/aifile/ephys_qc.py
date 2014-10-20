#!/usr/bin/python
import sys
import os
import h5py
import json
import numpy as np

TESTING = True

aifile = "157436.03.01.ai"

#if len(sys.argv) != 4:
#	if TESTING:
#		aifile = "157436.03.01.ai"
#	else:
#		print "Usage: %s <input ai> <input json> <output json>" % sys.argv[0]
#		sys.exit(1)
#else:
#	aifile = sys.argv[1]

if not os.path.isfile(aifile):
	print "Error: %s does not exist" % aifile
	sys.exit(1)

ai_file = h5py.File(aifile, "r")
#h5_file = h5py.File(infile, "r")

# pull out stimulus/response epochs
epochs = ai_file["epochs"]
lst = []
for k in epochs.keys():
	#k = k.encode('ascii', 'ignore')
	#print type(k)
	if k.startswith("Stim"):
		lst.append(int(k.split('_')[1]))
lst.sort()

sweeps = {}

# pull out Vm from beginning and end of recording
for i in range(len(lst)):
	results = {}
	labels = ""
	sweeps["Sweep_%d" % lst[i]] = results
	volts = epochs["Stim_%d" % lst[i]]["acq_voltage"]
	data = volts["sequence"]["data"].value
	print "Stim_%d" % lst[i]
	idx0a = volts["idx_start"].value
	idx0b = idx0a + int(0.025 / 5e-6)
	ival = 1000.0 * data[idx0a:idx0b]
	mean0 = np.mean(ival)
	ival -= mean0
	rms0 = np.sqrt(np.mean(np.square(ival)))
	peak0 = max(ival)
	peak0b = min(ival)
	if peak0 < -peak0b:
		peak0 = -peak0b
	results["vm_0"] = mean0
	results["noise_0"] = rms0
	print "\t%g\t%g\t%g" % (mean0, rms0, peak0)
	idx1b = volts["idx_stop"].value
	idx1a = idx1b - int(0.025 / 5e-6)
	ival = 1000.0 * data[idx1a:idx1b]
	mean1 = np.mean(ival)
	ival -= mean1
	rms1 = np.sqrt(np.mean(np.square(ival)))
	peak1 = max(ival)
	peak1b = min(ival)
	if peak1 < -peak1b:
		peak1 = -peak1b
	results["vm_1"] = mean1
	results["noise_1"] = rms1
	print "\t%g\t%g\t%g" % (mean1, rms1, peak1)
	if rms0 > 0.2:
		labels += "rms0 "
	if peak0 > 0.2:
		labels += "noise0 "
	if rms1 > 0.2:
		labels += "rms1 "
	if peak1 > 0.2:
		labels += "noise1 "
	if abs(mean1 - mean0) > 0.5:
		labels += "Vm "
	bridge = volts["sequence"]["bridge_balance"].value
	if bridge < 1e6:
		labels += "bridge-low "
	if bridge >20e6:
		labels += "bridge-high "
	print "\t%g" % bridge
	access = volts["sequence"]["access_resistance"].value
	print "\t%g" % access
	if len(labels) > 0:
		print "Sweep %d FAIL: %s" % (i, labels)

#h_acquisition = h_file["acquisition"]
#for k in h_acquisition.keys():
#	h_sweep = h_acquisition[k]
#	access_resistance = h_sweep["subclass"]["access_resistance"]
#	bridge_balance = h_sweep["subclass"]["bridge_balance"]

ai_file.close()


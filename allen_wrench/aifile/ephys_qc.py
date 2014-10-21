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

def get_first_epoch(idx0, stim):
	t0 = idx0
	t1 = t0 + int(0.25 / 5e-6)
	return t0, t1

def get_last_epoch(idx1, stim):
	return idx1-int(0.25/5e-6), idx1

def measure_vm(seg):
	vals = np.copy(seg)
	mean = np.mean(vals)
	vals -= mean
	rms = np.sqrt(np.mean(np.square(vals)))
	return mean, rms



# pull out Vm from beginning and end of recording
for i in range(len(lst)):
	results = {}
	labels = ""
	sweeps["Sweep_%d" % lst[i]] = results
	current = epochs["Stim_%d" % lst[i]]["stim_current"]["sequence"]["data"]
	volts = epochs["Stim_%d" % lst[i]]["acq_voltage"]
	data = volts["sequence"]["data"].value
	print "Stim_%d" % lst[i]
	# measure Vm and noise right before stimulus
	id0, id1 = get_first_epoch(volts["idx_start"].value, current)
	mean0, rms0 = measure_vm(1000 * data[id0:id1])
	results["vm_0"] = mean0
	results["rms_0"] = rms0
	# measure Vm and noise from end of recording
	id0, id1 = get_last_epoch(volts["idx_stop"].value, current)
	mean1, rms1 = measure_vm(1000 * data[id0:id1])
	results["vm_1"] = mean1
	results["rms_1"] = rms1
	# measure blowout voltage
	# take mean of V in S20_Blowout_DA_0
	templ = ai_file["stimulus"]["templates"]
	if "S20_Blowout_DA_0" not in templ:
		results["blowout"] = float('nan')
	else:
		swpname = templ["S20_Blowout_DA_0"].attrs["epochs"][0]
		curr = epochs[swpname]["stim_current"]["sequence"]["data"].value
		results["blowout"] = np.mean(curr)

	for k in results.keys():
		print "\t%s\t%g" % (k, results[k])

	if results["rms_0"] > 0.2:
		labels += "rms0 "
	if results["rms_1"] > 0.2:
		labels += "rms1 "
	if abs(results["vm_0"] - results["vm_1"]) > 0.5:
		labels += "Vm "
	bridge = volts["sequence"]["bridge_balance"].value
	if bridge < 1e6:
		labels += "bridge-low "
	if bridge >20e6:
		labels += "bridge-high "
	if results["blowout"] > 3.0:
		labels += "blowout "
#	print "\t%g" % bridge
#	access = volts["sequence"]["access_resistance"].value
#	print "\t%g" % access
	if len(labels) > 0:
		print "**\tSweep %d FAIL: %s\n" % (i, labels)

#h_acquisition = h_file["acquisition"]
#for k in h_acquisition.keys():
#	h_sweep = h_acquisition[k]
#	access_resistance = h_sweep["subclass"]["access_resistance"]
#	bridge_balance = h_sweep["subclass"]["bridge_balance"]

ai_file.close()


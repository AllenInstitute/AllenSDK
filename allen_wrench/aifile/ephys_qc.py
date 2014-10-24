#!/usr/bin/python
import sys
import os
import h5py
import json
import numpy as np

TESTING = True

path = "/local2/ephys/"
aifile = "157436.03.01.aibs"

if len(sys.argv) == 2:
	aifile = sys.argv[1]
	path = ""
#else:
#	print "Usage: %s <input ai>" % sys.argv[0]
#	sys.exit(1)

#if len(sys.argv) != 4:
#	if TESTING:
#		aifile = "157436.03.01.ai"
#	else:
#		print "Usage: %s <input ai> <input json> <output json>" % sys.argv[0]
#		sys.exit(1)
#else:
#	aifile = sys.argv[1]

if not os.path.isfile(path + aifile):
	print "Error: %s does not exist" % (path + aifile)
	sys.exit(1)

ai_file = h5py.File(path + aifile, "r")
#h5_file = h5py.File(infile, "r")


# pull out stimulus/response epochs
epochs = ai_file["epochs"]
lst = []
for k in epochs.keys():
	if k.startswith("Experiment"):
		lst.append(int(k.split('_')[1]))
lst.sort()

log_file = open(aifile + ".txt", "w")

sweeps = {}

def get_first_epoch(idx0, stim):
	t0 = idx0
	t1 = t0 + int(0.010 / 5e-6)
	return t0, t1

def get_last_epoch(idx1, stim):
	return idx1-int(0.010/5e-6), idx1

def measure_vm(seg):
	vals = np.copy(seg)
	mean = np.mean(vals)
	vals -= mean
	rms = np.sqrt(np.mean(np.square(vals)))
	return mean, rms

########################################################################
# experiment-level metrics

def measure_blowout():
	templ = ai_file["stimulus"]["templates"]
	blowout = None
	# fetch first epoch where blowout template was presented
	# blowout is average voltage response
	for k in templ.keys():
		if k.startswith("S20_Blowout"):
			swpname = templ[k].attrs["epochs"][0]
			swp_num = int(swpname.split('_')[1])
			swp = epochs["Experiment_%d" % swp_num]["response"]
			v = swp["sequence"]["data"].value
			idx0 = swp["idx_start"].value
			blowout = np.mean(v[idx0:])
			break
	return blowout


def measure_electrode_0():
	templ = ai_file["stimulus"]["templates"]
	e0 = None
	# fetch first epoch where in-bath template was presented
	# electrode 0 is average current response
	for k in templ.keys():
		if k.startswith("S2_InBath"):
			swpname = templ[k].attrs["epochs"][0]
			swp = epochs[swpname]["response"]
			curr = swp["sequence"]["data"].value
			idx0 = swp["idx_start"].value
			e0 = np.mean(curr[idx0:])
			break
	return 1e12 * e0

def measure_seal():
	templ = ai_file["stimulus"]["templates"]
	seal = None
	# fetch first epoch where cell attached template was presented
	# measure average of current response relative to applied voltage
	for k in templ.keys():
		if k.startswith("S4_CellAttached"):
			swpname = templ[k].attrs["epochs"][0]
			stim = epochs[swpname]["stimulus"]
			resp = epochs[swpname]["response"]
			# advance to end of pulse
			v = stim["sequence"]["data"].value
			curr = resp["sequence"]["data"].value
			for i in range(len(v)):
				if v[i] != 0:
					idx = i
					break
			# advance to end of pulse
			for i in range(idx, len(v)):
				if v[i] == 0:
					idx = i
					break
			# now step through each pulse and measure voltage in and
			#   average current out
			in_pulse = False
			start = i
			ohms = []
			base_i = 0
			for i in range(idx, len(v)):
				if in_pulse:
					if v[i] == 0:
						in_pulse = False
						# pulse over -- measure current over last half
						i1 = i - 1
						i0 = start - (i - start) / 2
						vmean = v[i-1]
						imean = np.mean(curr[i0:i1])
						imean -= base_i
						ohms.append(vmean / imean)
				else:
					if v[i] > 0:
						start = i
						in_pulse = True
						samps = int(0.01 / 5e-6)
						base_i = np.mean(curr[i-samps:i-1])
			seal = np.mean(ohms)
			break
	return seal

########################################################################
# experiment-level metrics

def analyze_voltage_clamp(num):
	# if InBath, measure Electrode 0
	# if S4, measure seal
	pass

def analyze_current_clamp(num):
	# find start of stimulus
	# get indices and measure Vm for 3 noise intervals
	pass


blowout = measure_blowout()
if blowout is not None:
	print "Blowout: %g mV" % blowout
e0 = measure_electrode_0()
if e0 is not None:
	print "Electrode 0: %g pA" % e0
seal = measure_seal()
if seal is not None:
	print "Seal: %e ohms" % seal

sys.exit(0)

# pull out Vm from beginning and end of recording
cnt = 0
for i in range(len(lst)):
	results = {}
	labels = ""
	sweeps["Sweep_%d" % lst[i]] = results
	print "Sweep_%d" % i
	current = epochs["Experiment_%d"%lst[i]]["stimulus"]["sequence"]["data"]
	volts = epochs["Experiment_%d" % lst[i]]["response"]
	data = volts["sequence"]["data"].value
	#print "Experiment_%d" % lst[i]
	log_file.write("---------- Sweep %d\n" % lst[i])
	# measure Vm and noise right before stimulus
	id0, id1 = get_first_epoch(volts["idx_start"].value, current)
	print "idx start: %d -> %d" % (id0, id1)
	mean0, rms0 = measure_vm(1000 * data[id0:id1])
	results["vm_0"] = mean0
	results["rms_0"] = rms0
	# measure Vm and noise from end of recording
	id0, id1 = get_last_epoch(volts["idx_stop"].value, current)
	print "idx end: %d -> %d" % (id0, id1)
	mean1, rms1 = measure_vm(1000 * data[id0:id1])
	results["vm_1"] = mean1
	results["rms_1"] = rms1
	# measure blowout voltage
	# take mean of V in S20_Blowout_DA_0
	templ = ai_file["stimulus"]["templates"]
	if "S20_Blowout_DA_0:0" not in templ:
		results["blowout"] = float('nan')
	else:
		swpname = templ["S20_Blowout_DA_0:0"].attrs["epochs"][0]
		curr = epochs[swpname]["response"]["sequence"]["data"].value
		results["blowout"] = np.mean(curr)

	for k in results.keys():
		#print "\t%s\t%g" % (k, results[k])
		log_file.write("\t%s\t%g\n" % (k, results[k]))

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
		#print "**\tSweep %d FAIL: %s\n" % (i, labels)
		log_file.write("**\tFAIL: %s\n" % labels)
		cnt += 1

print "%s: %d of %d sweeps passed" % (aifile, len(lst)-cnt, len(lst))
log_file.write("%d of %d sweeps passed\n" % (len(lst)-cnt, len(lst)))


#h_acquisition = h_file["acquisition"]
#for k in h_acquisition.keys():
#	h_sweep = h_acquisition[k]
#	access_resistance = h_sweep["subclass"]["access_resistance"]
#	bridge_balance = h_sweep["subclass"]["bridge_balance"]

ai_file.close()
log_file.close()


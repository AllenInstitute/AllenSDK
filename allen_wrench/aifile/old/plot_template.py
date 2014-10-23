#!/usr/bin/python
import h5py
import sys
import numpy as np
from matplotlib import pyplot as plt
from aif_common import list_templates

template_num = -1

infile = "157436.03.01.ai"
if len(sys.argv) != 2:
	print "usage: %s <group num>" % sys.argv[0]
	sys.exit(1)
else:
	template_num = int(sys.argv[1])

########################################################################
########################################################################
# helper functions

def scale_axis(ax):
	plt.sca(ax)
	limits = ax.axis()
	axdim = []
	axdim.append(limits[0])
	axdim.append(limits[1])
	axdim.append(limits[2])
	axdim.append(limits[3])
	if axdim[2] < 0:
		axdim[2] *= 1.05
	else:
		axdim[2] *= 0.95
	if axdim[3] < 0:
		axdim[3] *= 0.95
	else:
		axdim[3] *= 1.05
	plt.axis(axdim)


########################################################################
########################################################################
# fetch data from hdf5 file
f = h5py.File(infile, "r")

# get sweeps associated with specified template/group
template_dir, template_lst = list_templates(f)
if template_num < 0 or template_num >= len(template_lst):
	n = len(template_lst)
	print "Unable to read group %d. Only %d available" % (template_num, n)
	sys.exit(1)
swps = template_dir[template_lst[template_num]].attrs["epochs"]




## each sweep is stored in its own epoch
## fetch epoch for test pulse
#pulse_name = "Pulse_%d" % sweep_num
#if pulse_name not in f["epochs"]:
#	# some sweeps don't have a test pulse -- bail out in such cases
#	print "Specified sweep not available"
#	sys.exit(1)
#pulse_epoch = f["epochs"][pulse_name]
## get voltage stream 
#idx_0 = pulse_epoch["acquired"]["idx_start"].value
#idx_1 = pulse_epoch["acquired"]["idx_stop"].value
#pulse_v = pulse_epoch["acquired"]["sequence"]["data"][idx_0:idx_1]
#pulse_v *= 1000.0
## get current stream 
#idx_0 = pulse_epoch["stimulus"]["idx_start"].value
#idx_1 = pulse_epoch["stimulus"]["idx_stop"].value
#pulse_curr = pulse_epoch["stimulus"]["sequence"]["data"][idx_0:idx_1]
#pulse_curr *= 1.0e12
## get epoch's description
#pulse_name = pulse_epoch["description"].value
#
## generate time array, based on sampling rate
#srate = pulse_epoch["stimulus"]["sequence"]["sampling_rate"].value
#pulse_t = np.zeros(len(pulse_v))
#for i in range(len(pulse_t)):
#	pulse_t[i] = srate * i
#
## fetch epoch for experiment data
#stim_name = "Stim_%d" % sweep_num
#stim_epoch = f["epochs"][stim_name]
## get voltage stream 
#idx_0 = stim_epoch["acquired"]["idx_start"].value
#idx_1 = stim_epoch["acquired"]["idx_stop"].value
#stim_v = stim_epoch["acquired"]["sequence"]["data"][idx_0:idx_1]
#stim_v *= 1000.0
## get current stream 
#idx_0 = stim_epoch["stimulus"]["idx_start"].value
#idx_1 = stim_epoch["stimulus"]["idx_stop"].value
#stim_curr = stim_epoch["stimulus"]["sequence"]["data"][idx_0:idx_1]
#stim_curr *= 1e12
## get epoch's description
#stim_name = stim_epoch["description"].value
#
## generate time array, based on sampling rate
#stim_t = np.zeros(len(stim_v))
#for i in range(len(stim_t)):
#	stim_t[i] = srate * i
#

########################################################################
########################################################################
# plot data


########################################################################
# plotting
fig = plt.figure(figsize=(17,9), dpi=80)
fig.canvas.set_window_title(infile)
ax_v1 = plt.subplot2grid((2,3), (0,0), colspan=1)
ax_i1 = plt.subplot2grid((2,3), (1,0), colspan=1)
ax_v2 = plt.subplot2grid((2,3), (0,1), colspan=2)
ax_i2 = plt.subplot2grid((2,3), (1,1), colspan=2)

plt.sca(ax_v1)
plt.xlabel("Seconds")
plt.ylabel("Millivolts")
plt.sca(ax_i1)
plt.xlabel("Seconds")
plt.ylabel("Picoamps")
plt.sca(ax_v2)
plt.xlabel("Seconds")
plt.ylabel("Millivolts")
plt.sca(ax_i2)
plt.xlabel("Seconds")
plt.ylabel("Picoamps")

pulse_t = None
stim_t = None

for j in range(len(swps)):
	swp = swps[j]
	sweep_num = int(swp.split('_')[1])

	# each sweep is stored in its own epoch
	# fetch epoch for test pulse
	pulse_name = "TestPulse_%d" % sweep_num
	if pulse_name not in f["epochs"]:
		# some sweeps don't have a test pulse -- bail out in such cases
		print "Specified sweep not available"
		sys.exit(1)
	pulse_epoch = f["epochs"][pulse_name]
	# get voltage stream 
	idx_0 = pulse_epoch["acquired"]["idx_start"].value
	idx_1 = pulse_epoch["acquired"]["idx_stop"].value
	pulse_v = pulse_epoch["acquired"]["sequence"]["data"][idx_0:idx_1]
	pulse_v *= 1000.0
	# get current stream 
	idx_0 = pulse_epoch["stimulus"]["idx_start"].value
	idx_1 = pulse_epoch["stimulus"]["idx_stop"].value
	pulse_curr = pulse_epoch["stimulus"]["sequence"]["data"][idx_0:idx_1]
	pulse_curr *= 1.0e12
	# get epoch's description
	pulse_name = pulse_epoch["description"].value

	# generate time array, based on sampling rate
	if pulse_t is None:
		srate = pulse_epoch["stimulus"]["sequence"]["sampling_rate"].value
		pulse_t = np.zeros(len(pulse_v))
		for i in range(len(pulse_t)):
			pulse_t[i] = srate * i

	# fetch epoch for experiment data
	stim_name = "Stim_%d" % sweep_num
	stim_epoch = f["epochs"][stim_name]
	# get voltage stream 
	idx_0 = stim_epoch["acquired"]["idx_start"].value
	idx_1 = stim_epoch["acquired"]["idx_stop"].value
	stim_v = stim_epoch["acquired"]["sequence"]["data"][idx_0:idx_1]
	stim_v *= 1000.0
	# get current stream 
	idx_0 = stim_epoch["stimulus"]["idx_start"].value
	idx_1 = stim_epoch["stimulus"]["idx_stop"].value
	stim_curr = stim_epoch["stimulus"]["sequence"]["data"][idx_0:idx_1]
	stim_curr *= 1e12
	# get epoch's description
	stim_name = stim_epoch["description"].value

	if stim_t is None:
		# generate time array, based on sampling rate
		stim_t = np.zeros(len(stim_v))
		for i in range(len(stim_t)):
			stim_t[i] = srate * i

	#ax_v1.set_title(pulse_name)
	ax_v1.plot(pulse_t, pulse_v)
	ax_i1.plot(pulse_t, pulse_curr)

	#ax_v2.set_title(stim_name)
	ax_v2.plot(stim_t, stim_v, label="Sweep_%d"%sweep_num)
	ax_i2.plot(stim_t, stim_curr)

handles, labels = ax_v2.get_legend_handles_labels()
ax_v2.legend(handles, labels)

scale_axis(ax_i1)
scale_axis(ax_i2)
plt.draw()
plt.show()

f.close()


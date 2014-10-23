#!/usr/bin/python
import h5py
import sys
import numpy as np
from matplotlib import pyplot as plt

infile = "/local2/ephys/157615.04.02.ai"
if len(sys.argv) < 2:
	print "usage: %s <sweep num>" % sys.argv[0]
	sys.exit(1)
else:
	sweep_num = int(sys.argv[1])

########################################################################
########################################################################
# fetch data from hdf5 file
f = h5py.File(infile, "r")

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
srate = pulse_epoch["stimulus"]["sequence"]["sampling_rate"].value
pulse_t = np.zeros(len(pulse_v))
for i in range(len(pulse_t)):
	pulse_t[i] = srate * i

# fetch epoch for experiment data
stim_name = "Experiment_%d" % sweep_num
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

# generate time array, based on sampling rate
stim_t = np.zeros(len(stim_v))
for i in range(len(stim_t)):
	stim_t[i] = srate * i


########################################################################
########################################################################
# plot data

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

ax_v1.set_title(pulse_name)
ax_v1.plot(pulse_t, pulse_v)
ax_i1.plot(pulse_t, pulse_curr)
scale_axis(ax_i1)

ax_v2.set_title(stim_name)
ax_v2.plot(stim_t, stim_v)
ax_i2.plot(stim_t, stim_curr)
scale_axis(ax_i2)
plt.draw()
plt.show()

f.close()


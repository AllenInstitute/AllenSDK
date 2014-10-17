#!/usr/bin/python
import h5py
import sys
import numpy as np
from matplotlib import pyplot as plt

infile = "157436.03.01.ai"
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
# fetch epoch
epoch = f["epochs"]["Sweep_%d"%sweep_num]

# get voltage stream from epoch
idx_0 = epoch["acq_voltage"]["idx_start"].value
idx_1 = epoch["acq_voltage"]["idx_stop"].value
v = epoch["acq_voltage"]["sequence"]["data"][idx_0:idx_1]

# get current stream from epoch
idx_0 = epoch["stim_current"]["idx_start"].value
idx_1 = epoch["stim_current"]["idx_stop"].value
curr = epoch["stim_current"]["sequence"]["data"][idx_0:idx_1]

# get epoch's description
name = epoch["description"].value

# generate time array, based on sampling rate
srate = epoch["stim_current"]["sequence"]["sampling_rate"].value
t = np.zeros(len(v))
for i in range(len(t)):
	t[i] = srate * i

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

ax_v2.set_title(name)
#ax_v2.set_title("Sweep_%d" % sweep_num)
ax_v2.plot(t, v)
ax_i2.plot(t, curr)
scale_axis(ax_i2)
plt.draw()
plt.show()

f.close()


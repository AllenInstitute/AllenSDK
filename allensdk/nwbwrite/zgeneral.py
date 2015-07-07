#!/usr/bin/python
import sys
import numpy as np
import nwb
from nwbco import *

# create a new borg file. If we don't specify a time, the present time
#   will be used
fname = "sample_zgeneral.nwb"
borg = nwb.NWB(filename=fname, identifier="test",overwrite=True, description="Sample storing metadata")
print "Creating " + fname
borg.set_metadata_from_file("source_script", sys.argv[0])

borg.set_metadata("experimenter", "Genghis Khan")
borg.set_metadata_from_file("notes", __file__)

borg.set_metadata("custom_field", "African swallow")

borg.set_metadata(SLICES, "slice data")

borg.set_metadata(EXTRA_SHANK_DEVICE("shank1"), "recording device")
borg.set_metadata(EXTRA_SHANK_CUSTOM("shank1", "foo"), "custom field")
attrs = {}
attrs["reference_atlas"] = "AI Brain Atlas"
borg.set_metadata(EXTRA_SHANK_LOCATION("shank1"), "position x,y,z,...", x_coord="43.112", **attrs)
borg.set_metadata(EXTRA_IMPEDANCE, 1.0*np.arange(10))

borg.set_metadata(EXTRA_SHANK_DEVICE("shank1"), "Mary had a little lamb")

# when all data is entered, close the Borg file
borg.close()


#!/usr/bin/python
import sys
import numpy as np
import nwb

# create a new borg file. If we don't specify a time, the present time
#   will be used
borg = nwb.NWB(filename="sample_general.nwb", identifier="test",overwrite=True, description="Sample storing metadata")

borg.set_metadata("experimenter", "Genghis Khan")
borg.set_metadata_from_file("notes", __file__)

borg.set_metadata("custom_field", "African swallow")
borg.set_metadata_from_file("nwb.py", "nwb.py")

borg.set_metadata("extracellular_ephys/shank1/device", "recording device")
borg.set_metadata("extracellular_ephys/shank1/foo", "custom field")
borg.set_metadata("extracellular_ephys/impedance", 1.0*np.arange(10))

borg.set_metadata("intracellular_ephys/foo/device", "Mary had a little lamb")

# when all data is entered, close the Borg file
borg.close()


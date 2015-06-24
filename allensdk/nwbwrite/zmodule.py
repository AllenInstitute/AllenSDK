#!/usr/bin/python
import sys
import nwb

# create a new borg file. If we don't specify a time, the present time
#   will be used
borg = nwb.NWB(filename="sample_module.nwb", identifier="test", overwrite=True, description="Test file for simple module")

mod = borg.create_module("simple module")
mod.set_description("module description")
mod.finalize()

# when all data is entered, close the Borg file
borg.close()


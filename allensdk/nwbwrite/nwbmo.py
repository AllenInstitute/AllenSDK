import sys
import copy
import numpy as np
import traceback

class Module(object):
    """ Constructor for module

        Args:
            *name* (text) name of the module (must be unique for modality)

            *borg* (Borg object) Created borg file
    """
    def __init__(self, name, nwb, spec):
        self.name = name
        self.nwb = nwb
        self.spec = copy.deepcopy(spec)
        # a place to store interfaces belonging to this module
        self.ifaces = {}
        # create module folder immediately, so it's available 
        folder = self.borg.file_pointer["processing"]
        if name in folder:
            nwb.fatal_error("Module '%s' already exists" % name)
        self.mod_folder = folder.create_group(self.name)
        # 
        self.finalized = False

    def create_interface(self, iface_type, name):
        if iface_type == "ImageSegmentation":
            iface = ImageSegmentation(self, name)
        elif iface_type == "UnitTimes":
            iface = UnitTimes(self, name)
        else
            iface = Interface(self, name)
        self.ifaces[name] = iface

    def set_description(self, desc):
        """ Set description field in module

            Args:
                *desc* (text) Description of module

            Returns:
                *nothing*
        """
        self.set_value("description", desc)

    def set_source(self, src):
        """ Identify source(s) for the data provided in the module.
            This can be one or more other modules, or time series
            in acquisition or stimulus

            Args:
                *src* (text) Path to objects providing data that the
                data here is based on

            Returns:
                *nothing*
        """
        self.set_value("source", src)

    def set_value(self, key, value, **attrs):
        """Adds a custom key-value pair (ie, dataset) to the root of 
           the module.
   
           Args:
               *key* (string) A unique identifier within the TimeSeries

               *value* (any) The value associated with this key
   
           Returns:
               *nothing*
        """
        if self.finalized:
            nwb.fatal_error("Added value to module after finalization")
        self.spec[key] = copy.deepcopy(self.spec["[]"])
        dtype = self.spec[key]["_datatype"]
        name = "module " + self.name
        self.nwb.set_value_internal(key, value, self.spec, name, dtype, **attrs)

    def finalize(self):
        """ Completes the module and writes changes to disk.

            Args: 
                *none*

            Returns:
                *nothing*
        """
        if self.finalized:
            return
        self.finalized = True
        # make a string list of available interfaces
        ifaces = []
        # finalize interfaces
        for k in self.interface_dict.keys():
            self.interface_dict[k].finalize()
            ifaces.append(k)
        # add module attributes
        self.mod_folder.attrs["source"] = self.source
        self.mod_folder.attrs["description"] = self.description
        self.mod_folder.attrs["interfaces"] = ifaces
        self.mod_folder.attrs["neurodata_type"] = "Module"




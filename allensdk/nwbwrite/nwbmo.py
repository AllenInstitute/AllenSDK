import sys
import copy
import numpy as np
import traceback

class Module(object):
    """ Constructor for module

        Arguments:
            *name* (text) name of the module (must be unique)

            *nwb* (NWB object) Created nwb file

            *spec* (dict) dictionary structure defining module specification
    """
    def __init__(self, name, nwb, spec):
        self.name = name
        self.nwb = nwb
        self.spec = copy.deepcopy(spec)
        # a place to store interfaces belonging to this module
        self.ifaces = {}
        # create module folder immediately, so it's available 
        folder = self.nwb.file_pointer["processing"]
        if name in folder:
            nwb.fatal_error("Module '%s' already exists" % name)
        self.mod_folder = folder.create_group(self.name)
        # 
        self.finalized = False

    def full_path(self):
        """ returns HDF5 path of module
        """
        return "processing/" + self.name

    def create_interface(self, iface_type):
        """ Creates an interface within the module. 
            Each module can have multiple interfaces.
            Standard interface options are:

                BehavioralEpochs -- general container for storing and
                publishing intervals (IntervalSeries)

                BehavioralEvents -- general container for storing and
                publishing event series (TimeSeries)

                BehavioralTimeSeries -- general container for storing and
                publishing time series (TimeSeries)

                Clustering -- clustered spike data, whether from
                automatic clustering tools or as a result of manual
                sorting

                ClusterWaveform -- mean event waveform of clustered data

                CompassDirection -- publishes 1+ SpatialSeries storing
                direction in degrees (or radians) 

                DfOverF -- publishes 1+ RoiResponseSeries showing
                dF/F in observed ROIs

                EventDetection -- information about detected events

                EventWaveform -- publishes 1+ SpikeEventSeries
                of extracellularly recorded spike events

                EyeTracking -- publishes 1+ SpatialSeries storing 
                direction of gaze

                FeatureExtraction -- salient features of events

                FilteredEphys -- publishes 1+ ElectricalSeries storing
                data from digital filtering

                Fluorescence -- publishes 1+ RoiResponseSeries showing
                fluorescence of observed ROIs

                ImageSegmentation -- publishes groups of pixels that
                represent regions of interest in an image

                LFP -- a special case of FilteredEphys, filtered and
                downsampled for LFP signal

                MotionCorrection -- publishes image stacks whos frames
                have been corrected to account for motion

                Position -- publishes 1+ SpatialSeries storing physical
                position. This can be along x, xy or xyz axes

                PupilTracking -- publishes 1+ standard *TimeSeries* 
                that stores pupil size

                UnitTimes -- published data about the time(s) spikes
                were detected in an observed unit
        """
        if iface_type not in self.nwb.spec["Interface"]:
            self.nwb.fatal_error("unrecognized interface: " + iface_type)
        if_spec = self.create_interface_definition(iface_type)
        if iface_type == "ImageSegmentation":
            iface = ImageSegmentation(iface_type, self, if_spec)
        elif iface_type == "EventDetection":
            iface = EventDetection(iface_type, self, if_spec)
        elif iface_type == "Clustering":
            iface = Clustering(iface_type, self, if_spec)
        elif iface_type == "UnitTimes":
            iface = UnitTimes(iface_type, self, if_spec)
        elif iface_type == "MotionCorrection":
            iface = MotionCorrection(iface_type, self, if_spec)
        else:
            iface = Interface(iface_type, self, if_spec)
        self.ifaces[iface_type] = iface
        return iface

    # internal function
    # read spec to create time series definition. do it recursively 
    #   if time series are subclassed
    def create_interface_definition(self, if_type):
        super_spec = copy.deepcopy(self.nwb.spec["Interface"]["SuperInterface"])
        if_spec = self.nwb.spec["Interface"][if_type]
        import nwb
        return nwb.recursive_dictionary_merge(super_spec, if_spec)

    def set_description(self, desc):
        """ Set description field in module

            Arguments:
                *desc* (text) Description of module

            Returns:
                *nothing*
        """
        self.set_value("description", desc)

    def set_value(self, key, value, **attrs):
        """Adds a custom key-value pair (ie, dataset) to the root of 
           the module.
   
           Arguments:
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

            Arguments: 
                *none*

            Returns:
                *nothing*
        """
        if self.finalized:
            return
        self.finalized = True
        # finalize interfaces
        iface_names = []
        for k, v in self.ifaces.iteritems():
            v.finalize()
            iface_names.append(v.name)
        self.spec["_attributes"]["interfaces"]["_value"] = iface_names
        # write own data
        grp = self.nwb.file_pointer["processing/" + self.name]
        self.nwb.write_datasets(grp, "", self.spec)
        self.nwb.write_attributes(grp, self.spec)


class Interface(object):
    """ Constructor for Interface base class

        Arguments:
            
            *name* (text) name of interface (may be class name)
            
            *module* (*Module*) Reference to parent module object that
            interface belongs to

            *spec* (dict) dictionary structure defining module specification
    """
    def __init__(self, name, module, spec):
        self.module = module
        self.name = name
        self.nwb = module.nwb
        self.spec = copy.deepcopy(spec)
        # timeseries that are added to interface directly
        self.defined_timeseries = {}
        # timeseries that exist elsewhere and are HDF5-linked
        self.linked_timeseries = {}
        self.iface_folder = module.mod_folder.create_group(name)
        self.finalized = False

    def full_path(self):
        """ Returns the path to this interface
        """
        return "processing/" + self.module.name + "/" + self.name

    def add_timeseries(self, ts):
        """ Adds a *TimeSeries* to the interface, setting the storage path
            of the *TimeSeries*. When a *TimeSeries* is added to an
            interface, the interface manages storage and finalization
            of it
        """
        if self.finalized:
            self.nwb.fatal_error("Added value after finalization")
        if ts.name in self.defined_timeseries:
            self.nwb.fatal_error("time series %s already defined" % ts.name)
        if ts.name in self.linked_timeseries:
            self.nwb.fatal_error("time series %s already defined" % ts.name)
        self.defined_timeseries[ts.name] = ts.spec["_attributes"]["ancestry"]["_value"]
        ts.set_path("processing/" + self.module.name + "/" + self.name)
        ts.finalize()

    def add_timeseries_as_link(self, ts_name, path):
        """ Add a previously-defined *TimeSeries* to the interface
        """
        if self.finalized:
            self.nwb.fatal_error("Added value after finalization")
        if ts_name in self.defined_timeseries:
            self.nwb.fatal_error("time series %s already defined" % ts.name)
        if ts_name in self.linked_timeseries:
            self.nwb.fatal_error("time series %s already defined" % ts.name)
        self.linked_timeseries[ts_name] = path

    def set_source(self, src):
        """ Identify source(s) for the data provided in the module.
            This can be one or more other modules, or time series
            in acquisition or stimulus

            Arguments:
                *src* (text) Path to objects providing data that the
                data here is based on

            Returns:
                *nothing*
        """
        self.set_value("source", src)

    def set_value(self, key, value, **attrs):
        """Adds a custom key-value pair (ie, dataset) to the interface
   
           Arguments:
               *key* (string) A unique identifier within the TimeSeries

               *value* (any) The value associated with this key
   
           Returns:
               *nothing*
        """
        if self.finalized:
            self.nwb.fatal_error("Added value after finalization")
        if key not in self.spec:
            self.spec[key] = copy.deepcopy(self.spec["[]"])
        dtype = self.spec[key]["_datatype"]
        name = "module " + self.name
        self.nwb.set_value_internal(key, value, self.spec, name, dtype, **attrs)

    # TODO ensure mandatory values are present
    # have this an separate function that subclasses can override
    def ensure_mandatory(self):
        pass

    def finalize(self):
        """ Finish off the interface and write changes to disk
        """
        if self.finalized:
            return
        self.finalized = True
        # ensure mandatory values are present
        self.ensure_mandatory()
        # keep a running tally of errors and report all at end
        err_str = ""
        # verify that all required time series are present
        #   note: this requires searching each's ancestry list
        # allow time series that exist outside of those required
        if "_mandatory_timeseries" in self.spec:
            reqd = self.spec["_mandatory_timeseries"]
            for i in range(len(reqd)):
                tstype = reqd[i]
                match = False
                for name, ancestry in self.defined_timeseries.iteritems():
                    for j in range(len(ancestry)):
                        if tstype == ancestry[j]:
                            match = True
                            break
                    if match:
                        break
                # look for linked items
                if not match:
                    for name, path in self.linked_timeseries.iteritems():
                        tgt = self.nwb.file_pointer[path]
                        ancestry = tgt.attrs["ancestry"]
                        for j in range(len(ancestry)):
                            if tstype == ancestry[j]:
                                match = True
                                break
                        if match:
                            break
                if not match:
                    err_str += "Missing %s in interface %s\n" % (tstype, self.name)
        # check for mandatory fields
        for k, v in self.spec.iteritems():
            if k.startswith("_"):
                continue
            if k == "[]" or k == "<>":
                continue
            if v["_datatype"] == "group":
                continue
            if v["_include"] == "mandatory" and "_value" not in v:
                err_str += "Missing %s in interface %s\n" % (k, self.name)
        if len(err_str) > 0:
            self.nwb.fatal_error(err_str)
        # add a help string, using the interface description
        if "_description" in self.spec:
            helpdict = {}
            helpdict["_datatype"] = "str"
            helpdict["_value"] = self.spec["_description"]
            if "_attributes" not in self.spec:
                self.spec["_attributes"] = {}
            self.spec["_attributes"]["help"] = helpdict
        # write own data
        folder = "processing/" + self.module.name + "/" + self.name
        grp = self.nwb.file_pointer[folder]
        self.nwb.write_datasets(grp, "", self.spec)
        self.nwb.write_attributes(grp, self.spec)
        # create linked objects manually
        links = []
        for ts, path in self.linked_timeseries.iteritems():
            grp[ts] = self.nwb.file_pointer[path]
            links.append(str(ts + " => " + path))
        if len(links) > 0:
            grp.attrs["timeseries_links"] = links

########################################################################

class UnitTimes(Interface):
    def __init__(self, name, module, spec):
        super(UnitTimes, self).__init__(name, module, spec)
        self.unit_list = []
    
    def add_unit(self, unit_name, unit_times, description, source):
        """ Adds data about a unit to the module, including unit name,
            description and times. 

            Arguments:
                *unit_name* (text) Name of the unit, as it will appear in the file

                *unit_times* (double array) Times that the unit spiked

                *description* (text) Information about the unit

                *source* (text) Name, path or description of where unit times originated
        """
        if unit_name not in self.iface_folder:
            self.iface_folder.create_group(unit_name)
        else:
            self.nwb.fatal_error("unit %s already exists" % unit_name)
        spec = copy.deepcopy(self.spec["<>"])
        spec["unit_description"]["_value"] = description
        spec["times"]["_value"] = unit_times
        spec["source"]["_value"] = source
        self.spec[unit_name] = spec
        #unit_times = ut.create_dataset("times", data=unit_times, dtype='f8')
        #ut.create_dataset("unit_description", data=description)
        self.unit_list.append(str(unit_name))

    def append_unit_data(self, unit_name, key, value):
        """ Add auxiliary information (key-value) about a unit.
            Data will be stored in the folder that contains data
            about that unit.

            Arguments:
                *unit_name* (text) Name of unit, as it appears in the file

                *key* (text) Key under which the data is added

                *value* (any) Data to be added

            Returns:
                *nothing*
        """
        if unit_name not in self.spec:
            self.nwb.fatal_error("unrecognized unit name " + unit_name)
        spec = copy.deepcopy(self.spec["<>"]["[]"])
        spec["_value"] = value
        self.spec[unit_name][key] = spec
        #ut.create_dataset(data_name, data=aux_data)

    def finalize(self):
        """ Extended (subclassed) finalize procedure. It creates and stores a list of all units in the module and then
            calls the superclass finalizer.

            Arguments:
                *none*

            Returns:
                *nothing*
        """
        if self.finalized:
            return
        self.spec["unit_list"]["_value"] = self.unit_list
        if len(self.unit_list) == 0:
            self.nwb.fatal_error("UnitTimes interface created with no units")
        super(UnitTimes, self).finalize()

########################################################################

class Clustering(Interface):
    def finalize(self):
        if self.finalized:
            return
        # make record of which cluster numbers are used
        # make sure clusters are present
        if "_value" not in self.spec["num"]:
            self.nwb.fatal_error("Clustering module %s has no clusters" % self.full_path())
        # now make a list of unique clusters and sort them
        num_dict = {}
        num = self.spec["num"]["_value"]
        for i in range(len(num)):
            n = "%d" % num[i]
            if n not in num_dict:
                num_dict[n] = n
        num_array = []
        for k in num_dict.keys():
            num_array.append(int(k))
        num_array.sort()
        self.spec["cluster_nums"]["_value"] = num_array
        # continue with normal finalization
        super(Clustering, self).finalize()
    
########################################################################

class ImageSegmentation(Interface):
    def __init__(self, name, module, spec):
        super(ImageSegmentation, self).__init__(name, module, spec)
        # make a table to store what ROIs are added to which planes
        self.roi_list = {}

    def add_reference_image(self, plane, name, img):
        """ Add a reference image to the segmentation interface

            Arguments: 
                *plane* (text) name of imaging plane

                *name* (text) name of reference image

                *img* (byte array) raw pixel map of image, 8-bit grayscale

            Returns:
                *nothing*
        """
        img_ts = self.nwb.create_timeseries("ImageSeries", name)
        img_ts.set_value("format", "raw")
        img_ts.set_value("bits_per_pixel", 8)
        img_ts.set_value("dimension", [len(img[0]), len(img)])
        img_ts.set_time([0])
        img_ts.set_data(img, "grayscale", 1, 1)
        img_ts.set_path(self.full_path() + "/" + plane + "/reference_images/")
        img_ts.finalize()

    def add_reference_image_as_link(self, plane, name, path):
        # make sure path is valid
        if path not in self.nwb.file_pointer:
            self.nwb.fatal_error("Path '%s' not found in file" % path)
        # make sure target is actually a time series
        if self.nwb.file_pointer[path].attrs["neurodata_type"] != "TimeSeries":
            self.nwb.fatal_error("'%s' is not a TimeSeries" % path)
        # make sure plane is present
        if plane not in self.iface_folder:
            self.nwb.fatal_error("'%s' is not a defined imaging plane in %s" % (plane, self.full_path()))
        # create link
        grp = self.iface_folder[plane]["reference_images"]
        grp[name] = self.nwb.file_pointer[path]

    def create_imaging_plane(self, plane, description):
    #def create_imaging_plane(self, plane, manifold, reference_frame, meta_link):
        ''' Defines imaging manifold. This can be a simple 1D or
            2D manifold, a complex 3D manifold, or even random
            access. The manifold defines the spatial coordinates for
            each pixel. If multi-planar manifolds are to be defined
            separately, a separate imaging plane should be used for each.
            Non-planar manifolds should be stored as a vector.
            
            Pixels in the manifold must have a 1:1 correspondence
            with image segmentation masks and the masks and manifold
            must have the same dimensions.
        '''
        if plane not in self.spec:
            self.spec[plane] = copy.deepcopy(self.spec["<>"])
        #self.spec[plane]["manifold"]["_value"] = manifold
        #self.spec[plane]["reference_frame"]["_value"] = reference_frame
        self.spec[plane]["imaging_plane_name"]["_value"] = plane
        self.spec[plane]["description"]["_value"] = description
        grp = self.iface_folder.create_group(plane)
        grp.create_group("reference_images")
        self.roi_list[plane] = []

    def add_roi_mask_pixels(self, image_plane, roi_name, desc, pixel_list, weights, width, height):
        """ Adds an ROI to the module, with the ROI defined using a list of pixels.

            Arguments:
                *image_plane* (text) name of imaging plane
            
                *roi_name* (text) name of ROI

                *desc* (text) description of ROI

                *pixel_list* (2D int array) array of [x,y] pixel values

                *weights* (float array) array of pixel weights

                *width* (int) width of reference image, in pixels

                *height* (int) height of reference image, in pixels

            Returns:
                *nothing*
        """
        # create image out of pixel list
        img = np.zeros((height, width))
        for i in range(len(pixel_list)):
            y = pixel_list[i][0]
            x = pixel_list[i][1]
            img[y][x] = weights[i]
        self.add_masks(image_plane, roi_name, desc, pixel_list, weights, img)

    def add_roi_mask_img(self, image_plane, roi_name, desc, img):
        """ Adds an ROI to the module, with the ROI defined within a 2D image.

            Arguments:
                *image_plane* (text) name of imaging plane

                *roi_name* (text) name of ROI

                *desc* (text) description of ROI

                *img* (2D float array) description of ROI in a pixel map (float[y][x])

            Returns:
                *nothing*
        """
        # create pixel list out of image
        pixel_list = []
        for y in range(len(img)):
            row = img[y]
            for x in range(len(row)):
                if row[x] != 0:
                    pixel_list.append([x, y])
                    weights.append(row[x])
        self.add_masks(image_plane, name, pixel_list, weights, img)

    # internal function
    def add_masks(self, plane, name, desc, pixel_list, weights, img):
        if plane not in self.spec:
            self.nwb.fatal_error("Imaging plane %s not defined" % plane)
        if name in self.spec[plane]:
            self.nwb.fatal_error("Imaging plane %s already has ROI %s" % (plane, name))
        self.spec[plane][name] = copy.deepcopy(self.spec["<>"]["<>"])
        self.spec[plane][name]["pix_mask"]["_value"] = pixel_list
        self.spec[plane][name]["pix_mask"]["_attributes"]["weight"]["_value"] = weights
        self.spec[plane][name]["img_mask"]["_value"] = img
        self.spec[plane][name]["roi_description"]["_value"] = desc
        self.roi_list[plane].append(name)

    def finalize(self):
        if self.finalized:
            return
        # create roi_list for each plane
        for plane, roi_list in self.roi_list.iteritems():
            self.spec[plane]["roi_list"]["_value"] = roi_list
        # continue with normal finalization
        super(ImageSegmentation, self).finalize()

class MotionCorrection(Interface):
    def add_corrected_image(self, name, orig, xy_translation, corrected):
        """ Adds a motion-corrected image to the module, including
            the original image stack, the x,y delta necessary to
            shift the image frames for registration, and the corrected
            image stack.
            NOTE 1: All 3 timeseries use the same timestamps and so can share/
            link timestamp arrays

            NOTE 2: The timeseries passed in as 'xy_translation' and
            'corrected' will be renamed to these names, if they are not
            links to existing timeseries

            NOTE 3: The timeseries arguments can be either TimeSeries
            objects (new or old in case of latter 2 args) or strings.
            If they are new TimeSeries objects, they will be stored
            within the module. If they are existing objects, a link
            to those objects will be created

            Arguments:
                *orig* (ImageSeries or text) ImageSeries object or
                text path to original image time series

                *xy_translation* TimeSeries storing displacements of
                x and y direction in the data[] field

                *corrected* Motion-corrected ImageSeries

            Returns:
                *nothing*
        """
        self.spec[name] = copy.deepcopy(self.spec["<>"])
        # create link to original object
        import nwbts
        if isinstance(orig, nwbts.TimeSeries):
            if not orig.finalized:
                self.nwb.fatal_error("Original timeseries must already be stored and finalized")
            orig_path = orig.full_path()
        else:
            orig_path = orig
        self.spec[name]["original"]["_value_hardlink"] = orig_path
        links = "'original' is '%s'" % orig_path
        # finalize other time series, or create link if a string was
        #   provided
        # XY translation
        if isinstance(xy_translation, (str, unicode)):
            self.spec[name]["xy_translation"]["_value_hardlink"] = xy_translation
            links += "; 'xy_translation' is '%s'" % xy_translation
        else:
            if xy_translation.finalized:
                # timeseries exists in file -- link to it
                self.spec[name]["xy_translation"]["_value_hardlink"] = xy_translation.full_path()
                links += "; 'corrected' is '%s'" % xy_translation.full_path()
            else:
                # new timeseries -- set it's path and finalize it
                xy_translation.set_path("processing/" + self.module.name + "/MotionCorrection/" + name + "/")
                xy_translation.reset_name("xy_translation")
                xy_translation.finalize()
        # corrected series
        if isinstance(corrected, (str, unicode)):
            self.spec[name]["corrected"]["_value_hardlink"] = corrected
            links += "; 'corrected' is '%s'" % corrected
        else:
            if corrected.finalized:
                # timeseries exists in file -- link to it
                self.spec[name]["corrected"]["_value_hardlink"] = corrected.full_path()
                links += "; 'corrected' is '%s'" % corrected.full_path()
            else:
                corrected.set_path("processing/" + self.module.name + "/MotionCorrection/" + name + "/")
                corrected.reset_name("corrected")
                corrected.finalize()
        self.spec[name]["_attributes"]["links"]["_value"] = links
        import nwb
        nwb.write_json("foo.json", self.spec)


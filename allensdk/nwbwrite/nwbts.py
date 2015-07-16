import sys
import traceback
import h5py
import copy
import collections
import numpy as np
import nwbmo

class TimeSeries(object):
    """ Standard TimeSeries constructor. This is executed indirectly
        by calls to NWB.create_timeseries(). It should not be called
        directly.
    """
    def __init__(self, name, modality, spec, nwb):
        self.name = name
        # make a local copy of the specification, one that can be modified
        self.spec = copy.deepcopy(spec)
        # file handling
        self.nwb = nwb
        self.finalized = False
        # check modality and set path
        if modality == "acquisition":
            self.path = "/acquisition/timeseries/"
        elif modality == "stimulus":
            self.path = "/stimulus/presentation/"
        elif modality == "template":
            self.path = "/stimulus/templates/"
        elif modality == "other":
            self.path = ""
        else:
            m = "Modality must be acquisition, stimulus, template or other"
            self.fatal_error(m)
        if self.path is not None:
            full_path = self.path + self.name
            if full_path in self.nwb.file_pointer:
                self.fatal_error("group '%s' already exists" % full_path)

    # internal function
    def fatal_error(self, msg):
        print "Error: " + msg
        print "TimeSeries: " + self.name
        print "Stack trace follows"
        print "-------------------"
        traceback.print_stack()
        sys.exit(1)

    def reset_name(self, name):
        """ Change the name of a *TimeSeries*
        """
        self.name = name

    ####################################################################
    # set field values

    # don't allow setting attributes on values, not for now at least
    # it's not legal to add attributes to fields that are in the spec as
    #   there is no way to mark them as custom
    def set_value(self, key, value, dtype=None):
        """ Set key-value pair, with optional attributes (in dtype)
        """
        if self.finalized:
            self.fatal_error("Added value after finalization")
        name = "TimeSeries %s" % self.name
        self.nwb.set_value_internal(key, value, self.spec, name, dtype)

    # add a link to another NWB object
    def set_value_as_link(self, key, value):
        """ Create a link to another NWB object
        """
        if self.finalized:
            self.fatal_error("Added value after finalization")
        # check type
        path = ""
        if isinstance(value, TimeSeries):
            path = value.full_path()
        elif isinstance(value, nwbmo.Interface):
            path = value.full_path()
        elif isinstance(value, (str, unicode)):
            path = value
        else:
            self.fatal_error("Unrecognized type for setting up link -- found %s" % type(value))
        self.set_value(key, path)
        self.set_value(key + "_path", path)

    # add a link to another NWB object
    def set_value_as_remote_link(self, key, target_file, dataset_path):
        """ FINISH ME
        """
        assert False, "FINISH ME"
        if self.finalized:
            self.fatal_error("Added value after finalization")
        # check type
        if not isinstance(target_file, (str, unicode)):
            self.fatal_error("File name must be string, received %s", type(target_file))
        if not isinstance(dataset_path, (str, unicode)):
            self.fatal_error("Dataset path must be string, received %s", type(dataset_path))
        # TODO set _value_softlink fields
        set_value(key, "????")
        set_value(key + "_link", target_file + "::" + dataset_path)

    # internal function used for setting data[] and timestamps[]
    # this method doesn't include necessary logic to manage attributes
    #   and prevent the user from adding custom attributes to
    #   standard fields, or to alter required 'custom' attribute
    def set_value_with_attributes_internal(self, key, value, dtype, **attrs):
        if self.finalized:
            self.fatal_error("Added value after finalization")
        name = "TimeSeries %s" % self.name
        self.nwb.set_value_internal(key, value, self.spec, name, dtype, **attrs)

    # have special calls for those that are common to all time series
    def set_description(self, value):
        """ Convenience function to set the description field of the
            *TimeSeries*
        """
        if self.finalized:
            self.fatal_error("Added value after finalization")
        self.spec["_attributes"]["description"]["_value"] = str(value)

    def set_comments(self, value):
        """ Convenience function to set the comments field of the
            *TimeSeries*
        """
        if self.finalized:
            self.fatal_error("Added value after finalization")
        self.spec["_attributes"]["comments"]["_value"] = str(value)

    # for backward compatibility to screwy scripts, and to be nice
    #   in event of typo
    def set_comment(self, value):
        self.set_comments(value)

    def set_source(self, value):
        """ Convenience function to set the source field of the
            *TimeSeries*
        """
        if self.finalized:
            self.fatal_error("Added value after finalization")
        self.spec["_attributes"]["source"]["_value"] = str(value)

    def set_time(self, timearray):
        ''' Store timestamps for the time series. 
   
           Arguments:
               *timearray* (double array) Timestamps for each element in *data*
   
           Returns:
               *nothing*
        '''
        # t_interval should have default value set to 1 in spec file
        self.set_value("timestamps", timearray)

    def set_time_by_rate(self, time_zero, rate):
        '''Store time by start time and sampling rate only
   
           Arguments:
               *time_zero* (double) Time of data[] start. For template stimuli, this should be zero
               *rate* (float) Cycles per second (Hz)
   
           Returns:
               *nothing*
        '''
        attrs = {}
        attrs["rate"] = rate
        self.set_value_with_attributes_internal("starting_time", time_zero, None, **attrs)

    def ignore_time(self):
        """ In some cases (eg, template stimuli) there is no time 
            data available. Rather than store invalid data, it's better
            to explicitly avoid setting the time field

            Arguments:
                *none*

            Returns:
                *nothing*
        """
        if self.finalized:
            self.fatal_error("Changed timeseries after finalization")
        # downgrade required status so file will generate w/o
        self.spec["timestamps"]["_include"] = "standard"
        self.spec["starting_time"]["_include"] = "standard"
        self.spec["num_samples"]["_include"] = "standard"

    # if default value used, value taken from specification file
    def set_data(self, data, units=None, conversion=None, resolution=None, dtype=None):
        '''Defines the data stored in the TimeSeries. Type of data 
           depends on which class of TimeSeries is being used

           Arguments:
               *data* (user-defined) Array of data samples stored in time series

               *units* (text) Base SI unit for data[] (eg, Amps, Volts)

               *conversion* (float) Multiplier necessary to convert elements in data[] to specified unit

               *resolution* (float) Minimum meaningful distance between elements in data[] (e.g., the +/- range, quantal step size between values, etc)
   
           Returns:
               *nothing*
        '''
        attrs = {}
        if units is not None:
            attrs["units"] = str(units)
        if conversion is not None:
            attrs["conversion"] = float(conversion)
        if resolution is not None:
            attrs["resolution"] = float(resolution)
        self.set_value_with_attributes_internal("data", data, dtype, **attrs)

    def ignore_data(self):
        """ In some cases (eg, externally stored image files) there is no 
            data to be stored. Rather than store invalid data, it's better
            to explicitly avoid setting the data field

            Arguments:
                *none*

            Returns:
                *nothing*
        """
        if self.finalized:
            self.fatal_error("Changed timeseries after finalization")
        # downgrade required status so file will generate w/o
        self.spec["data"]["_include"] = "standard"

    ####################################################################
    ####################################################################
    # linking code

    # takes the path to the sibling time series to create a hard link
    #   to its timestamp array
    def set_time_as_link(self, sibling):
        '''Links the *timestamps* dataset in this TimeSeries to that
           stored in another TimeSeries. This is useful when multiple time
           series have data that is recorded on the same clock.
           This works by making an HDF5 hard link to the timestamps array
           in the sibling time series
   
           Arguments:
               *sibling* (text) Full HDF5 path to TimeSeries containing source timestamps array, or a python TimeSeries object
   
           Returns:
               *nothing*
        '''
        tgt_path = self.create_hardlink("timestamps", sibling)
        # tell kernel about link so table of all links can be added to
        #   file at end
        self.nwb.record_timeseries_time_link(self.full_path(), tgt_path)


    def set_data_as_link(self, sibling):
        '''Links the *data* dataset in this TimeSeries to that
           stored in another TimeSeries. This is useful when multiple time
           series represent the same data.
           This works by making an HDF5 hard link to the data array
           in the sibling time series
   
           Arguments:
               *sibling* (text) Full HDF5 path to TimeSeries containing source data[] array, or a python TimeSeries object
   
           Returns:
               *nothing*
        '''
        if self.finalized:
            self.fatal_error("Added value after finalization")
        tgt_path = self.create_hardlink("data", sibling)
        # tell kernel about link so table of all links can be added to
        #   file at end
        self.nwb.record_timeseries_data_link(self.full_path(), tgt_path)


    def set_data_as_remote_link(self, file_path, dataset_path):
        '''Links the *data* dataset in this TimeSeries to data stored
           in an external file, using and HDF5 soft-link.
           The dataset in the external file must contain attributes required 
           for the TimeSeries::data[] element.
   
           Arguments:
               *file_path* (text) File-system path to remote HDF5 file

               *dataset_path* (text) Full path within remote HDF5 file to dataset
   
           Returns:
               *nothing*
        '''
        if self.finalized:
            self.fatal_error("Added value after finalization")
        self.create_softlink("data", file_path, dataset_path)
        self.nwb.record_timeseries_data_soft_link(self.full_path(), file_path+"://"+dataset_path)


    # internal function
    # creates link to similarly named field between two groups
    def create_hardlink(self, field, target):
        # type safety -- make sure sibling is class if not string
        if self.finalized:
            self.fatal_error("Added value after finalization")
        if isinstance(target, str):
            sib_path = target
        elif isinstance(target, TimeSeries):
            sib_path = target.full_path()
        elif isinstance(target, nwbmo.Module):
            sib_path = target.full_path()
        else:
            self.fatal_error("Unrecognized link-to object. Expected str or TimeSeries, found %s" % type(target))
        # define link. throw error if value was already set
        if "_value" in self.spec[field]:
            self.fatal_error("cannot specify a link after setting value")
        elif "_value_softlink" in self.spec[field]:
            self.fatal_error("cannot specify both hard and soft links")
        self.spec[field]["_value_hardlink"] = sib_path + "/" + field
        # return path string
        return sib_path


    # internal function
    # creates link to similarly named field between two groups
    def create_softlink(self, field, file_path, dataset_path):
        if self.finalized:
            self.fatal_error("Added value after finalization")
        if "_value" in self.spec[field]:
            self.fatal_error("cannot specify a data link after set_data()")
        elif "_value_hardlink" in self.spec[field]:
            self.fatal_error("cannot specify both hard and soft links")
        self.spec[field]["_value_softlink"] = dataset_path
        self.spec[field]["_value_softlink_file"] = file_path


    ####################################################################
    ####################################################################
    # file writing and path management

    def set_path(self, path):
        """ Sets the path for where the *TimeSeries* is created. This
            is only necessary for *TimeSeries* that were not created
            indicating the common storage location (ie, acquisition,
            stimulus, template) or that were not added to a module
        """
        if self.finalized:
            self.fatal_error("Added value after finalization")
        if path.endswith('/'):
            self.path = path
        else:
            self.path = path + "/"
        full_path = self.path + self.name
        if full_path in self.nwb.file_pointer:
            self.fatal_error("group '%s' already exists" % full_path)

    def full_path(self):
        """Returns the HDF5 path to this *TimeSeries*
        """
        return self.path + self.name

    def finalize(self):
        """ Finish the *TimeSeries* and write changes to disk
        """
        if self.finalized:
            return
        # verify all mandatory fields are present
        spec = self.spec
        # num_samples can sometimes be calculated automatically. do so
        #   here if that's possible
        if "_value" not in spec["num_samples"]:
            if "_value" in spec["timestamps"]:
                # make tmp short name to avoid passing 80-col limit in editor
                tdat = spec["timestamps"] 
                spec["num_samples"]["_value"] = len(tdat["_value"])
        # document missing required fields
        err_str = []
        for k in spec.keys():
            if k.startswith('_'):   # check for leading underscore
                continue    # control field -- ignore
            if "_value" in spec[k]:
                continue    # field exists
            if spec[k]["_include"] == "required":
                # value is missing -- see if alternate or link exists
                if "_value_softlink" in spec[k]:
                    continue
                if "_value_hardlink" in spec[k]:
                    continue
                # valid alternative fields include links -- check possibilities
                if "_alternative" in spec[k]:
                    if "_value" in spec[spec[k]["_alternative"]]:
                        continue    # alternative field exists
                    if "_value_hardlink" in spec[spec[k]["_alternative"]]:
                        continue    # alternative field exists
                    if "_value_softlink" in spec[spec[k]["_alternative"]]:
                        continue    # alternative field exists
                miss_str = "Missing field '%s'" % k
                if "_alternative" in spec[k]:
                    miss_str += " (or '%s')" % spec[k]["_alternative"]
                err_str.append(str(miss_str))
            # make a record of missing required fields
            if spec[k]["_include"] == "standard":
                if "_value" not in spec["_attributes"]["missing_fields"]:
                    spec["_attributes"]["missing_fields"]["_value"] = []
                spec["_attributes"]["missing_fields"]["_value"].append(str(k))
        # add spec's _description to 'help' field
        # use asserts -- there's a problem w/ the spec if these don't exist
        assert "help" in spec["_attributes"]
        assert "_description" in spec
        spec["_attributes"]["help"]["_value"] = spec["_description"]
        # make sure that mandatory attributes are present
        for k in spec["_attributes"]:
            if spec["_attributes"][k]["_include"] == "required":
                if "_value" not in spec["_attributes"][k]:
                    err_str.append("Missing attribute " + k)
        # report errors for missing mandatory data
        if len(err_str) > 0:
            print "TimeSeries creation error (name=%s)" % self.name
            if len(err_str) == 1:
                print "Missing mandatory field:"
            else:
                print "Missing mandatory field(s):"
            for i in range(len(err_str)):
                print "\t" + err_str[i]
            sys.exit(1)
        # TODO validate path (not None and is valid)

        # TODO check _linkto

        # write content to file
        grp = self.nwb.file_pointer.create_group(self.path + self.name)
        self.nwb.write_datasets(grp, "", spec)
        self.nwb.write_attributes(grp, spec)

        # allow freeing of memory
        self.spec = None
        # set done flag
        self.finalized = True


class AnnotationSeries(TimeSeries):
    '''Stores text-based records about the experiment. To use the
       AnnotationSeries, add records individually through 
       add_annotation() and then call finalize(). Alternatively, if 
       all annotations are already stored in a list, use set_data()
       and set_timestamps()

       **Constructor arguments:**
           *name*    name of the time series (must be unique for modality)

           *modality*    "acquisition", "stimulus", "template" or "other"

           *nwb*    NWB object

       Returns:
           *nothing*
    ''' 
    def __init__(self, name, modality, spec, nwb):
        super(AnnotationSeries, self).__init__(name, modality, spec, nwb)
        self.annot_str = []
        self.annot_time = []

    def add_annotation(self, what, when):
        '''Conveninece function to add annotations individually

        Arguments:
            *what* (text) Annotation

            *when* (double) Timestamp for annotation

        Returns:
            *nothing*
        '''
        self.annot_str.append(str(what))
        self.annot_time.append(float(when))

    def finalize(self):
        '''Extends superclass call by pushing annotations onto 
        the data[] and timestamps[] fields

        Arguments:
            *none*

        Returns:
            *nothing*
        '''
        if self.finalized:
            return
        if len(self.annot_str) > 0:
            if "_value" in self.spec["data"]:
                print "AnnotationSeries error -- can only call set_data() or add_annotation(), not both"
                print "AnnotationSeries name: " + self.name
                sys.exit(1)
            if "_value" in self.spec["timestamps"]:
                print "AnnotationSeries error -- can only call set_time() or add_annotation(), not both"
                print "AnnotationSeries name: " + self.name
                sys.exit(1)
            self.spec["data"]["_value"] = self.annot_str
            self.spec["timestamps"]["_value"] = self.annot_time
        super(AnnotationSeries, self).finalize()


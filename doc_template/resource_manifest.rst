Resource Manifest
=================

Large or complex models and experiments generally need more than
a single :doc:`model description file </model_description>`
to completely describe an experiment.  A manifest file is a way to
describe all of the resources needed within
the Allen Wrench description format itself.

The manifest section is named "manifest" by default,
though it is configurable.  The objects in the manifest section
each specify a directory, file, or file pattern.
Files and directories may be organized in a parent-child relationship.

A Simple Manifest
-----------------

This is a simple manifest file that specifies the BASEDIR directory
using ".", meaning the current directory:
::

    {
        "manifest": [
            {   "key": "BASEDIR",
                "type": "dir",
                "spec": "."
            }
        ] }
    }

Parent Child Relationships
--------------------------

Adding the optional "parent_key" member to a manifest object
creates a parent-child relation.  In this case WORKDIR will
be found in "./work":
::

    {
        "manifest": [
            {   "key": "BASEDIR",
                "type": "dir",
                "spec": "."
            },
            {   "key": "WORKDIR",
                "type": "dir",
                "spec": "/work",
                "parent_key": "BASEDIR"
            }
        ] }
    }

File Spec Patterns
------------------

Files can be specified using the type "file" instead of "dir".
If a sequence of many files is needed, the spec may contain patterns
to indicate where the sequence number (%d) or string (%s) will be
interpolated:
::

    {
        "manifest": [
            {   "key": "BASEDIR",
                "type": "dir",
                "spec": "."
            },
            {
                "key": "voltage_out_cell_path",
                "type": "file",
                "spec": "v_out-cell-%d.dat",
                "parent_key": "BASEDIR"
            }
        ] }
    }


.. _lob_in_manifest:

Large Object References
-----------------------

A common use for a manifest
is to reference a :doc:`large object file </large_object_parser>`
from a model description.  This is done by specifying a "format" in
the manifest object.  If the format is one that is recognized by an
internal Allen Wrench large object parser or one that is user-provided,
the data from the large object can be accessed from the description object
at simulation time.

::

    {
        "manifest": [
            {   "key": "BASEDIR",
                "type": "dir",
                "spec": "."
            },
            {
                "key": "positions_path",
                "type": "file",
                "format": "hdf5",
                "spec": "positions.h5",
                "parent_key": "WORKDIR"
            }
        ] }
    }
    
While large object parsers are designed to work
with manifest files, there's no reason that a manifest could be use
without a LobParser if the simulation software provides
an alternate way of accessing the external file.  Similarly,
it is possible to pass a filename directly to a large object file
without using a manifest.

Split Manifest Files
--------------------

manifest files can be split like any description file.
This allows the specification of a general directory structure in a
shared file and specific files in a separate configuration
(i.e. stimulus and working directory)


Extensions
----------

To date, manifest description files have not been used to reference
URLs that provide model data, but it is a planned future use case.


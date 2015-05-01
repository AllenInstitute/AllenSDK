Resource Manifest
=================

Large or complex models and experiments generally need more than
a single :doc:`model description file </model_description>`
to completely describe an experiment.  A manifest file is a way to
describe all of the resources needed within
the Allen SDK description format itself.

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


Split Manifest Files
--------------------

Manifest files can be split like any description file.
This allows the specification of a general directory structure in a
shared file and specific files in a separate configuration
(i.e. stimulus and working directory)


Extensions
----------

To date, manifest description files have not been used to reference
URLs that provide model data, but it is a planned future use case.


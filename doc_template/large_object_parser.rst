Large Object Parser
===================

The :doc:`model description </model_description>` JSON format provided by Allen Wrench
is not ideal for representing large or specialized data.
Examples include large connection matrices, electrophysiology stimulus data,
image or video data to be used as stimulus, or even small data that is already 
available in another format from another source, such as a comma-separated-values (CSV)
file exported from a spreadsheet.
The Allen Wrench library provides a large object parser (LOBParser) interface
to handle such files.


Large Objects, Manifests and Descriptions
-----------------------------------------

Large object files can be accessed from a model description 
by using the "format" :ref:`field of a manifest object <lob_in_manifest>`.

HDF5 LOB Parser
---------------


ORCA LOB Parser
---------------


Pandas LOB Parser
-----------------
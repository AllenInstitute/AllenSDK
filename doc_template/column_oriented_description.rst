Large Models and Special Situations
===================================

Models with large data structures can be difficult to represent in JSON.
Often it is best to use a file format better suited to the data and connect it to the
:doc:`json description </model_description>` 
using an alternative format with a resource manifest.
The following techniques might also help.


Column-Oriented Data
--------------------
    While there is no explicit support for column-oriented data in this format,
    a column-oriented structure can be encoded to avoid repeating the names of the
    structure members.
    
    ::
    
        {
            "column_oriented" [
               { "section": "positions",
                 "data" { "x": [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ],
                          "y": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                          "z": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] }
            ]
        }
        
    The conventions to follow are:
    
    #. The section name is "column_oriented" (configurable).
    #. Each object within the column oriented section specifies a separate table.
    #. The "section" field of a table object acts as a section name when the description is read.
    #. The remaining fields of the table object are the members of the structure.
    #. The values are arrays with an entry for each object in the section.
    #. All of the arrays must be the same length.

Stream-able Split Descriptions
------------------------------

    For ease of use with streaming systems, the following modifications should be made
    to a split description file.
    
    #. The top level braces, section name and array brackets must be ommitted.
    #. The section name must be known implicitly or be provided externally.
    #. Each object within the section must be on a single line.
    #. Objects are separated by a carriage return following the comma.
    #. Only one section should be present in each stream-able file.

    cells_big_streamable.json:
    
    ::
    
    { "name": "cell1", "position": [0.1, 0.2, 0.3], "type": "pyramidal", "Vr": -2.0 },
    { "name": "cell2", "position": [0.5, 0.3, 0.1], "type": "pyramidal", "Vr": -1.6 },
    ...etc...
    ::
    
    { "name": "cell100000", "position": [0.5, 0.3, 0.1], "type": "pyramidal", "Vr": -1.6 }
    
    This allows the streaming system to still use the Allen SDK library
    to access the individual objects without having to load the entire description into memory.
    
    Column-oriented sections should not be stored as stream-able description files.
    Instead use an alternate format that natively supports column-oriented data 
    such as pandas or HDF5.
  
  
Python Configuration Parser
---------------------------

Instead of storing a model description as JSON, it can be stored as
a serialized (not pickled) Python structure.
All of the same conventions and recomendations about the structure of the
file still apply (top level sections containing arrays of objects,
shallow nesting, arrays are ok, literal values).

A model description stored this way uses the .pycfg extension by convention.
values are also allowed to be Python expressions,
including calls to built-in functions, the numpy library and the csa library.

This version of the model description format is not recommended for data exchange as it
requires a Python interpreter to be parsed.  It is recommended if the model can best be
described using matrix operations.
It is a compromise between a purely structural model description
and a procedural scripting language.


Model Description API
---------------------

A model description can simply be built in memory using a Python script.
This approach is best for developing tools that would generate an Allen SDK configuration,
either from scratch or from another format.  It can also be used where complex transformations
to the data are needed to generate the model that aren't available in the JSON format.

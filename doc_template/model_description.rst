Model Description
=================

Basic Structure
---------------

    A model description file is simply a JSON object with several sections at the top level
    and an array of JSON objects within each section.
    
    ::
    
            {
               "cell_section": [
                   { 
                     "name": "cell 1",
                     "shape": "pyramidal"
                     "position": [ 0.1, 0.2, 0.3 ]
                   },
                   {
                     "name": "cell 2",
                     "shape": "glial",
                     "position": [ 0.1, 0.2, 0.3 ]
                   }
               ],
               "extra": [
                  { "what": "wood",
                    "who": "woodchuck"
                  }
               ]
           }
   
    Even if a section contains no objects or only one object the array brackets must be present.
    
    
Objects Within Sections
-----------------------

    While no restrictions are enforced on what kinds of objects are stored in a section,
    some rules of thumb make the file easier to work with.
    
    #. All objects within a section are the same structure.
       Common operations on a section are to display it as a table,
       iterate over it, load from or write to a spreadsheet or csv file.
       These operations are all easier if the section is fairly homogenous.
    #. Objects are not deeply nested.
       While some shallow nesting is often useful, deep nesting such as a tree structure
       is not recommended.
       It makes interoperability with other tools and data formats more difficult.
    #. Arrays are ok, though they should not be deeply nested either.
    #. Object member values should be literals.  Do not use pickeled classes, for example.

Comments
--------

    The JSON specification does not allow comments.
    However, the Allen Wrench library applies a preprocessing stage
    to remove C++-style comments, so they can be used in description files.
    
    Multi-line comments should be surrounded by /* */
    and single-line comments start with //.
    Commented description files will not be recognized by strict json parsers
    unless the comments are stripped.
    
    commented.json:
    ::
        {
           /*
            *  multi-line comment
            */
           "section1": [
               {
                  "name": "simon"  // single line comment
               }]
           }

Split Description Files by Section
----------------------------------

    A model description can be split into multiple files
    by putting some sections in one file and other sections into another file.
    This can be useful if you want to put a topology of cells and connections in one file
    and experimental conditions and stimulus in another file.  The resulting structure in
    memory will behave the same way as if the files were not split.
    This allows a small experiment to be described in a single file
    and large experiments to be more modular.

    cells.json:
    ::
    
            {
               "cell_section": [
                   { 
                     "name": "cell 1",
                     "shape": "pyramidal"
                     "position": [ 0.1, 0.2, 0.3 ]
                   },
                   {
                     "name": "cell 2",
                     "shape": "glial",
                     "position": [ 0.1, 0.2, 0.3 ]
                   }
               ]
           }
    
    extras.json:
    ::
    
           {
               "extra": [
                  { 
                    "what": "wood",
                    "who": "woodchuck"
                  }
               ]
           }
           

Split Sections Between Description Files
----------------------------------------

If two description files containing the same sections are combined,
the resulting description will contain objects from both files.
This feature allows sub-networks to be described in separate files.
The sub-networks can then be composed into a larger network with an additional
description of the interconnections.

    network1.json:
    ::
        /* A self-contained sub-network */
        {
            "cells": [
                { "name": "cell1" },
                { "name": "cell2" }
            ],
            /* intra-network connections /*
            "connections": [
                { "source": "cell1", "target" : "cell2" }
            ]
        }
    
    network2.json:
    ::
        /* Another self-contained sub-network */
        {
            "cells": [
                { "name": "cell3" },
                { "name": "cell4" }
            ],
            "connections": [
                { "source": "cell3", "target" : "cell4" }
            ]
        }
    
    interconnect.json:
    ::
    
        {
            // the additional connections needed to
            // connect the network1 and network2
            // into a ring topology.
            "connections": [
               { "source": "cell2", "target": "cell3" },
               { "source": "cell4", "target": "cell1" }
            ]
        }

Manifests, Large Data and Special Situations
--------------------------------------------

JSON has many advantages.  It is widely supported, readable and easy to parse and edit.
As data sets get larger or specialized those advantages diminish.
The Allen Wrench library has several mechanisms for handling parts of the model description
that are not easily expressable as JSON. 
:doc:`Large object parsers </large_object_parser>` can be used to create an interface
between the model description and the large data files.  They can be thought of as a thin
translation layer.  The :doc:`resource manifest </resource_manifest>`
is a way of storing the file and directory locations in a model description.
Some :doc:`techniques </column_oriented_description>` to make the JSON more compatible
with big data technologies are also available in specific instances
without resorting to external formats.


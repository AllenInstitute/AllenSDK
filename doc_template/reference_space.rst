Reference Space
===============

Allen Institute atlases and data are registered, when possible, to one of several common reference spaces. Working in such a space allows you to 
easily compare data across subjects and experimental modalities.

This page documents how to use the Allen SDK to interact with a reference space. For more information and a list of reference spaces, see the 
`API documentation <http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies>`_.


Structure Tree
--------------

Brain structures in our reference spaces are arranged in trees. The leaf nodes of the tree describe the very fine anatomical divisions 
of the space, while nodes closer to the root correspond to gross divisions. The :py:class:`~allensdk.core.structure_tree.StructureTree` 
class provides an interface for interacting with a structure tree. 


To download a structure tree, use the :py:class:`allensdk.api.queries.ontologies_api.OntologiesApi` class as seen in 
`this example <_static/examples/nb/reference_space.html#Constructing-a-structure-tree>`_


Annotation Volumes
------------------

An annotation volume is a 3d raster image that segments the reference space into structures. Each voxel in the annotation volume is assigned 
an integer value that describes the finest structure to which that point in space definitely belongs. 

To download a nrrd formatted annotation volume at a specified isometric resolution, use the :py:class:`allensdk.api.queries.mouse_connectivity_api` class. 
There is `an example <_static/examples/nb/reference_space.html#Downloading-an-annotation-volume>`_ in the notebook.


ReferenceSpace Class
---------------------

The :py:class:`allensdk.core.reference_space.ReferenceSpace` class contains methods for working with our reference spaces. Some use cases might include:

    - `Building an indicator mask for one or more structures <_static/examples/nb/reference_space.html#making structure masks>`_ 
    - `Viewing the annotation <_static/exampled/nb/reference_space.html#View a slice from the annotation>`_
    - `Querying the structure graph <_static/exampled/nb/reference_space.html#Using a StructureTree>`_
    
Please see the `example notebook <_static/examples/nb/reference_space.html>`_ for more code samples.


# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import pytest

import os


@pytest.mark.nightly
def test_notebook(tmpdir_factory):

    # coding: utf-8

    # # Reference Space
    #
    # This notebook contains example code demonstrating the use of the StructureTree and ReferenceSpace classes. These classes provide methods for interacting with the 3d spaces to which Allen Institute data and atlases are registered.
    #
    # Unlike the AllenSDK cache classes, StructureTree and ReferenceSpace operate entirely in memory. We recommend using json files to store text and nrrd files to store volumetric images.
    #
    # The MouseConnectivityCache class has methods for downloading, storing, and constructing StructureTrees and ReferenceSpaces. Please see [here](https://alleninstitute.github.io/AllenSDK/_static/examples/nb/mouse_connectivity.html) for examples.

    # ## Constructing a StructureTree
    #
    # A StructureTree object is a wrapper around a structure graph - a list of dictionaries documenting brain structures and their containment relationships. To build a structure tree, you will first need to obtain a structure graph.
    #
    # For a list of atlases and corresponding structure graph ids, see [here](http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies).

    # In[1]:

    from allensdk.api.queries.ontologies_api import OntologiesApi
    from allensdk.core.structure_tree import StructureTree

    oapi = OntologiesApi()
    structure_graph = oapi.get_structures_with_sets([1])  # 1 is the id of the adult mouse structure graph

    # This removes some unused fields returned by the query
    structure_graph = StructureTree.clean_structures(structure_graph)  

    tree = StructureTree(structure_graph)


    # In[2]:

    # now let's take a look at a structure
    tree.get_structures_by_name(['Dorsal auditory area'])


    # The fields are:
    #     * acronym: a shortened name for the structure
    #     * rgb_triplet: each structure is assigned a consistent color for visualizations
    #     * graph_id: the structure graph to which this structure belongs
    #     * graph_order: each structure is assigned a consistent position in the flattened graph
    #     * id: a unique integer identifier
    #     * name: the full name of the structure
    #     * structure_id_path: traces a path from the root node of the tree to this structure
    #     * structure_set_ids: the structure belongs to these predefined groups

    # ## Using a StructureTree

    # In[3]:

    # get a structure's parent
    tree.parent([1011])


    # In[4]:

    # get a dictionary mapping structure ids to names

    name_map = tree.get_name_map()
    name_map[247]


    # In[5]:

    # ask whether one structure is contained within another

    strida = 385
    stridb = 247

    is_desc = '' if tree.structure_descends_from(385, 247) else ' not'

    print( '{0} is{1} in {2}'.format(name_map[strida], is_desc, name_map[stridb]) )


    # In[6]:

    # build a custom map that looks up acronyms by ids
    # the syntax here is just a pair of node-wise functions. 
    # The first one returns keys while the second one returns values

    acronym_map = tree.value_map(lambda x: x['id'], lambda y: y['acronym'])
    print( acronym_map[385] )


    # ## Downloading an annotation volume
    #
    # This code snippet will download and store a nrrd file containing the Allen Common Coordinate Framework annotation. We have requested an annotation with 25-micron isometric spacing. The orientation of this space is:
    #     * Anterior -> Posterior
    #     * Superior -> Inferior
    #     * Left -> Right
    # This is the no-frills way to download an annotation volume. See the <a href='_static/examples/nb/mouse_connectivity.html#Manipulating-Grid-Data'>mouse connectivity</a> examples if you want to properly cache the downloaded data.

    # In[7]:

    import os
    import nrrd
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
    from allensdk.config.manifest import Manifest

    # the annotation download writes a file, so we will need somwhere to put it
    annotation_dir = str(tmpdir_factory.mktemp('annotation'))

    annotation_path = os.path.join(annotation_dir, 'annotation.nrrd')

    mcapi = MouseConnectivityApi()
    mcapi.download_annotation_volume('annotation/ccf_2016', 25, annotation_path)

    annotation, meta = nrrd.read(annotation_path)


    # ## Constructing a ReferenceSpace

    # In[8]:

    from allensdk.core.reference_space import ReferenceSpace

    # build a reference space from a StructureTree and annotation volume, the third argument is 
    # the resolution of the space in microns
    rsp = ReferenceSpace(tree, annotation, [25, 25, 25])


    # ## Using a ReferenceSpace

    # #### making structure masks
    #
    # The simplest use of a Reference space is to build binary indicator masks for structures or groups of structures.

    # In[9]:

    # A complete mask for one structure
    whole_cortex_mask = rsp.make_structure_mask([315])

    # view in coronal section


    # What if you want a mask for a whole collection of ontologically disparate structures? Just pass more structure ids to make_structure_masks:

    # In[10]:

    # This gets all of the structures targeted by the Allen Brain Observatory project
    brain_observatory_structures = rsp.structure_tree.get_structures_by_set_id([514166994])
    brain_observatory_ids = [st['id'] for st in brain_observatory_structures]

    brain_observatory_mask = rsp.make_structure_mask(brain_observatory_ids)

    # view in horizontal section

    # You can also make and store a number of structure_masks at once:

    # In[11]:

    import functools

    # Define a wrapper function that will control the mask generation. 
    # This one checks for a nrrd file in the specified base directory 
    # and builds/writes the mask only if one does not exist
    mask_writer = functools.partial(ReferenceSpace.check_and_write, annotation_dir)
        
    # many_structure_masks is a generator - nothing has actrually been run yet
    mask_generator = rsp.many_structure_masks([385, 1097], mask_writer)

    # consume the resulting iterator to make and write the masks
    for structure_id in mask_generator:
        print( 'made mask for structure {0}.'.format(structure_id) ) 

    os.listdir(annotation_dir)


    # #### Removing unassigned structures

    # A structure graph may contain structures that are not used in a particular reference space. Having these around can complicate use of the reference space, so we generally want to remove them.
    #
    # We'll try this using "Somatosensory areas, layer 6a" as a test case. In the 2016 ccf space, this structure is unused in favor of finer distinctions (e.g. "Primary somatosensory area, barrel field, layer 6a").

    # In[12]:

    # Double-check the voxel counts
    no_voxel_id = rsp.structure_tree.get_structures_by_name(['Somatosensory areas, layer 6a'])[0]['id']
    print( 'voxel count for structure {0}: {1}'.format(no_voxel_id, rsp.total_voxel_map[no_voxel_id]) )

    # remove unassigned structures from the ReferenceSpace's StructureTree
    rsp.remove_unassigned()

    # check the structure tree
    no_voxel_id in rsp.structure_tree.node_ids()


    # #### View a slice from the annotation

    # In[13]:

    import numpy as np


    # #### Downsample the space
    #
    # If you want an annotation at a resolution we don't provide, you can make one with the downsample method.

    # In[14]:

    import warnings

    target_resolution = [75, 75, 75]

    # in some versions of scipy, scipy.ndimage.zoom raises a helpful but distracting 
    # warning about the method used to truncate integers. 
    warnings.simplefilter('ignore')

    sf_rsp = rsp.downsample(target_resolution)

    # re-enable warnings
    warnings.simplefilter('default')

    print( rsp.annotation.shape )
    print( sf_rsp.annotation.shape )


    # Now view the downsampled space:

    # In[15]:


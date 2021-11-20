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

    # ## Mouse Connectivity
    # 
    # This notebook demonstrates how to access and manipulate data in the Allen Mouse Brain Connectivity Atlas. The `MouseConnectivityCache` AllenSDK class provides methods for downloading metadata about experiments, including their viral injection site and the mouse's transgenic line. You can request information either as a Pandas DataFrame or a simple list of dictionaries.
    # 
    # An important feature of the `MouseConnectivityCache` is how it stores and retrieves data for you. By default, it will create (or read) a manifest file that keeps track of where various connectivity atlas data are stored. If you request something that has not already been downloaded, it will download it and store it in a well known location.
    # 
    # Download this notebook in .ipynb format <a href='mouse_connectivity.ipynb'>here</a>.

    # In[1]:

    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

    # The manifest file is a simple JSON file that keeps track of all of
    # the data that has already been downloaded onto the hard drives.
    # If you supply a relative path, it is assumed to be relative to your
    # current working directory.
    manifest_file = tmpdir_factory.mktemp('mcc').join('manifest.json')
    mcc = MouseConnectivityCache(manifest_file=str(manifest_file))

    # open up a list of all of the experiments
    all_experiments = mcc.get_experiments(dataframe=True)
    print("%d total experiments" % len(all_experiments))

    # take a look at what we know about an experiment with a primary motor injection
    all_experiments.loc[122642490]


    # `MouseConnectivityCache` has a method for retrieving the adult mouse structure tree as an `StructureTree` class instance. This is a wrapper around a list of dictionaries, where each dictionary describes a structure. It is principally useful for looking up structures by their properties.

    # In[2]:

    # pandas for nice tables
    import pandas as pd

    # grab the StructureTree instance
    structure_tree = mcc.get_structure_tree()

    # get info on some structures
    structures = structure_tree.get_structures_by_name(['Primary visual area', 'Hypothalamus'])
    pd.DataFrame(structures)


    # As a convenience, structures are grouped in to named collections called "structure sets". These sets can be used to quickly gather a useful subset of structures from the tree. The criteria used to define structure sets are eclectic; a structure set might list:
    # 
    # * structures that were used in a particular project.
    # * structures that coarsely partition the brain.
    # * structures that bear functional similarity.
    # 
    # or something else entirely. To view all of the available structure sets along with their descriptions, follow this [link](http://api.brain-map.org/api/v2/data/StructureSet/query.json). To see only structure sets relevant to the adult mouse brain, use the StructureTree:

    # In[3]:

    from allensdk.api.queries.ontologies_api import OntologiesApi

    oapi = OntologiesApi()

    # get the ids of all the structure sets in the tree
    structure_set_ids = structure_tree.get_structure_sets()

    # query the API for information on those structure sets
    pd.DataFrame(oapi.get_structure_sets(structure_set_ids))


    # On the connectivity atlas web site, you'll see that we show most of our data at a fairly coarse structure level. We did this by creating a structure set of ~300 structures, which we call the "summary structures". We can use the structure tree to get all of the structures in this set:

    # In[4]:

    # From the above table, "Mouse Connectivity - Summary" has id 167587189
    summary_structures = structure_tree.get_structures_by_set_id([167587189])
    pd.DataFrame(summary_structures)


    # This is how you can filter experiments by transgenic line:

    # In[5]:

    # fetch the experiments that have injections in the isocortex of cre-positive mice
    isocortex = structure_tree.get_structures_by_name(['Isocortex'])[0]
    cre_cortical_experiments = mcc.get_experiments(cre=True, 
                                                    injection_structure_ids=[isocortex['id']])

    print("%d cre cortical experiments" % len(cre_cortical_experiments))

    # same as before, but restrict the cre line
    rbp4_cortical_experiments = mcc.get_experiments(cre=[ 'Rbp4-Cre_KL100' ], 
                                                    injection_structure_ids=[isocortex['id']])


    print("%d Rbp4 cortical experiments" % len(rbp4_cortical_experiments))


    # ## Structure Signal Unionization
    # 
    # The ProjectionStructureUnionizes API data tells you how much signal there was in a given structure and experiment. It contains the density of projecting signal, volume of projecting signal, and other information. `MouseConnectivityCache` provides methods for querying and storing this data.

    # In[6]:

    # find wild-type injections into primary visual area
    visp = structure_tree.get_structures_by_acronym(['VISp'])[0]
    visp_experiments = mcc.get_experiments(cre=False, 
                                           injection_structure_ids=[visp['id']])

    print("%d VISp experiments" % len(visp_experiments))

    structure_unionizes = mcc.get_structure_unionizes([ e['id'] for e in visp_experiments ], 
                                                      is_injection=False,
                                                      structure_ids=[isocortex['id']],
                                                      include_descendants=True)

    print("%d VISp non-injection, cortical structure unionizes" % len(structure_unionizes))


    # In[7]:

    structure_unionizes.head()


    # This is a rather large table, even for a relatively small number of experiments.  You can filter it down to a smaller list of structures like this.

    # In[8]:

    dense_unionizes = structure_unionizes[ structure_unionizes.projection_density > .5 ]
    large_unionizes = dense_unionizes[ dense_unionizes.volume > .5 ]
    large_structures = pd.DataFrame(structure_tree.nodes(large_unionizes.structure_id))

    print("%d large, dense, cortical, non-injection unionizes, %d structures" % ( len(large_unionizes), len(large_structures) ))

    print(large_structures.name)

    large_unionizes


    # ## Generating a Projection Matrix
    # The `MouseConnectivityCache` class provides a helper method for converting ProjectionStructureUnionize records for a set of experiments and structures into a matrix.  This code snippet demonstrates how to make a matrix of projection density values in auditory sub-structures for cre-negative VISp experiments. 

    # In[9]:

    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    visp_experiment_ids = [ e['id'] for e in visp_experiments ]
    ctx_children = structure_tree.child_ids( [isocortex['id']] )[0]

    pm = mcc.get_projection_matrix(experiment_ids = visp_experiment_ids, 
                                   projection_structure_ids = ctx_children,
                                   hemisphere_ids= [2], # right hemisphere, ipsilateral
                                   parameter = 'projection_density')

    row_labels = pm['rows'] # these are just experiment ids
    column_labels = [ c['label'] for c in pm['columns'] ] 
    matrix = pm['matrix']

    fig, ax = plt.subplots(figsize=(15,15))
    heatmap = ax.pcolor(matrix, cmap=plt.cm.afmhot)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(matrix.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(matrix.shape[0])+0.5, minor=False)

    ax.set_xlim([0, matrix.shape[1]])
    ax.set_ylim([0, matrix.shape[0]])          

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(column_labels, minor=False)
    ax.set_yticklabels(row_labels, minor=False)

    # ## Manipulating Grid Data
    # 
    # The `MouseConnectivityCache` class also helps you download and open every experiment's projection grid data volume. By default it will download 25um volumes, but you could also download data at other resolutions if you prefer (10um, 50um, 100um).
    # 
    # This demonstrates how you can load the projection density for a particular experiment. It also shows how to download the template volume to which all grid data is registered. Voxels in that template have been structurally annotated by neuroanatomists and the resulting labels stored in a separate annotation volume image.

    # In[10]:

    # we'll take this experiment - an injection into the primary somatosensory - as an example
    experiment_id = 181599674


    # In[11]:

    # projection density: number of projecting pixels / voxel volume
    pd, pd_info = mcc.get_projection_density(experiment_id)

    # injection density: number of projecting pixels in injection site / voxel volume
    ind, ind_info = mcc.get_injection_density(experiment_id)

    # injection fraction: number of pixels in injection site / voxel volume
    inf, inf_info = mcc.get_injection_fraction(experiment_id)

    # data mask:
    # binary mask indicating which voxels contain valid data
    dm, dm_info = mcc.get_data_mask(experiment_id)

    template, template_info = mcc.get_template_volume()
    annot, annot_info = mcc.get_annotation_volume()

    # in addition to the annotation volume, you can get binary masks for individual structures
    # in this case, we'll get one for the isocortex
    cortex_mask, cm_info = mcc.get_structure_mask(315)

    print(pd_info)
    print(pd.shape, template.shape, annot.shape)


    # Once you have these loaded, you can use matplotlib see what they look like.

    # In[12]:

    # compute the maximum intensity projection (along the anterior-posterior axis) of the projection data
    pd_mip = pd.max(axis=0)
    ind_mip = ind.max(axis=0)
    inf_mip = inf.max(axis=0)

    # show that slice of all volumes side-by-side
    f, pr_axes = plt.subplots(1, 3, figsize=(15, 6))

    pr_axes[0].imshow(pd_mip, cmap='hot', aspect='equal')
    pr_axes[0].set_title("projection density MaxIP")

    pr_axes[1].imshow(ind_mip, cmap='hot', aspect='equal')
    pr_axes[1].set_title("injection density MaxIP")

    pr_axes[2].imshow(inf_mip, cmap='hot', aspect='equal')
    pr_axes[2].set_title("injection fraction MaxIP")


    # In[13]:

    # Look at a slice from the average template and annotation volumes

    # pick a slice to show
    slice_idx = 264

    f, ccf_axes = plt.subplots(1, 3, figsize=(15, 6))

    ccf_axes[0].imshow(template[slice_idx,:,:], cmap='gray', aspect='equal', vmin=template.min(), vmax=template.max())
    ccf_axes[0].set_title("registration template")

    ccf_axes[1].imshow(annot[slice_idx,:,:], cmap='gray', aspect='equal', vmin=0, vmax=2000)
    ccf_axes[1].set_title("annotation volume")

    ccf_axes[2].imshow(cortex_mask[slice_idx,:,:], cmap='gray', aspect='equal', vmin=0, vmax=1)
    ccf_axes[2].set_title("isocortex mask")


    # On occasion the TissueCyte microscope fails to acquire a tile. In this case the data from that tile should not be used for analysis. The data mask associated with each experiment can be used to determine which portions of the grid data came from correctly acquired tiles.
    # 
    # In this experiment, a missed tile can be seen in the data mask as a dark warped square. The values in the mask exist within [0, 1], describing the fraction of each voxel that was correctly acquired

    # In[14]:

    f, data_mask_axis = plt.subplots(figsize=(5, 6))

    data_mask_axis.imshow(dm[81, :, :], cmap='hot', aspect='equal', vmin=0, vmax=1)
    data_mask_axis.set_title('data mask')



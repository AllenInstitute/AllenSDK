Connectivity Grid Data
----------------------

Image data from each Allen Mouse Connectivity Atlas experiment is 
registered to the Common Coordinate Framework (CCF). This produces a series of 
CCF-space 3D "grid" volumes which describe the location and extent of that experiment's 
injection site (containing infected cell bodies) as well as the distribution of 
AAV Projection signal throughout the brain.

You can access these grid volumes either `via the API <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data>`_
or by constructing a :py:class:`allensdk.core.mouse_connectivity_cache.MouseConnectivityCache`.

We produce grid volumes from registered images using the `grid module <https://github.com/AllenInstitute/AllenSDK/tree/master/allensdk/mouse_connectivity/grid>`_. 
These volumes are then used downstream by the `unionize module <https://github.com/AllenInstitute/AllenSDK/tree/master/allensdk/internal/mouse_connectivity/interval_unionize>`_, which
summarizes AAV signal within each annotated brain structure, producing :doc:`structure unionizes </unionizes>`.


Reference
---------

========================= ===========================================
Volume                    Description                                
========================= ===========================================
projection_density        sum of detected projection pixels / sum of all pixels in voxel
injection_fraction        fraction of pixels belonging to manually annotated injection site
injection_density         density of detected projection pixels within the manually annotated injection site
data_mask                 binary mask indicating if a voxel contains valid data. Only valid voxels should be used for analysis.
========================= ===========================================

Note that projection density reports detected AAV signal both within and outside the injection site.

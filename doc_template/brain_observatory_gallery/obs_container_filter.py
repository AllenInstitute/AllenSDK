"""
Find Experiment Containers using Metadata
=========================================

Find experiment containers that have specific targeted structures, cre lines, and imaging depths.
"""

####################################################################################
# This class uses a `manifest` to keep track of downloaded data and metadata.  
# All downloaded files will be stored relative to the directory holding the manifest
# file.  If `manifest_file` is a relative path (as it is below), it will be 
# saved relative to your working directory.  It can also be an absolute path.

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

#############################################################
# Next we'll download the list of all possible targeted areas
# and imaging depths to help decide what to filter for

targeted_structures = boc.get_all_targeted_structures()
print("all targeted structures: " + str(targeted_structures))

depths = boc.get_all_imaging_depths()
print("all imaging depths: " + str(depths))

cre_lines = boc.get_all_cre_lines()
print("all cre lines: " + str(cre_lines))

####################################################################
# Download experiment containers for VISp that are 300-335um deep.

import pprint
ecs = boc.get_experiment_containers(targeted_structures=['VISp'], 
                                    imaging_depths=[300,320,325,335])
print("VISp, 300-335um deep experiment containers: %d" % len(ecs))
print("Example experiment container record:")
pprint.pprint(ecs[0])


######################################################
# Download experiment containers for VISrl with Cux2 mice.

ecs = boc.get_experiment_containers(targeted_structures=['VISrl'], 
                                    cre_lines=['Cux2-CreERT2'])
print("VISrl, Cux2 experiment containers: %d" % len(ecs))
print("Example experiment container record:")
pprint.pprint(ecs[0])

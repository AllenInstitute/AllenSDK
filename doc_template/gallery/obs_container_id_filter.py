"""
Find Containers and Sessions by ID
==================================

Find experiment containers and experiment sessions by ID.
"""

####################################################################################
# This class uses a `manifest` to keep track of downloaded data and metadata.  
# All downloaded files will be stored relative to the directory holding the manifest
# file.  If `manifest_file` is a relative path (as it is below), it will be 
# saved relative to your working directory.  It can also be an absolute path.

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

# If you know the ID going id, you can use it for filtering.
# Note that inputs and returns are both lists!
ecs = boc.get_experiment_containers(ids=[580051757])
pprint.pprint(ecs[0])

# The same syntax works for experiment sessions.
exps = boc.get_ophys_experiments(ids=[580051759])
pprint.pprint(exps[0])



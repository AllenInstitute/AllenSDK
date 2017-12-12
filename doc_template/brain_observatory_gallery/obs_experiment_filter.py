"""
Find Experiment Sessions by Stimulus
====================================

Find experiment sessions for a specific stimulus.
"""

####################################################################################
# This class uses a `manifest` to keep track of downloaded data and metadata.  
# All downloaded files will be stored relative to the directory holding the manifest
# file.  If `manifest_file` is a relative path (as it is below), it will be 
# saved relative to your working directory.  It can also be an absolute path.

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

#############################################################
# Next we'll download the list of all possible Cre lines
# and stimuli to help decide what to filter for.

stimuli = boc.get_all_stimuli()
print("all stimuli: " + str(stimuli))

cre_lines = boc.get_all_cre_lines()
print("all Cre lines: " + str(cre_lines))

####################################################################
# Thes stimulus strings have constants in the `stimulus_info` module
# that can help avoid typos. Now let's find all experiments that 
# have the natural movie three stimulus.

import allensdk.brain_observatory.stimulus_info as si
import pprint

exps = boc.get_ophys_experiments(stimuli=[si.NATURAL_MOVIE_THREE])
print("natural_movie_three experiments: %d" % len(exps))
print("Example experiment record:")
pprint.pprint(exps[0])

####################################################################
# We can also combine filters to find all Rbp4 experiment sessinos
# with the drifting gratings stimulus.

exps = boc.get_ophys_experiments(stimuli=[si.DRIFTING_GRATINGS],
                                 cre_lines=['Rbp4-Cre_KL100'])
print("Drifting graMatching experiments: %d" % len(exps))
print("Example experiment record:")
pprint.pprint(exps[0])

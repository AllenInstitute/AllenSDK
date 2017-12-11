r'''

Receptive Field Analysis
========================
This example walks you through the capabilities of the receptive_field_analysis subpackage.
This package uses neuronal responses to the locally sparse noise stimulus in order to find 
on and off spatial receptive fields for visual cortex neurons. Also included are tools for 
visualizing the results of these analyses.

 We highly recommend reading through the 
[stimulus analysis whitepaper](http://help.brain-map.org/display/observatory/Documentation)
to understand the locally sparse noise stimulus and the analysis methodology.

'''


###############################################################################
# We will need the following packages and modules

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.receptive_field_analysis.visualization as rfvis
import allensdk.brain_observatory.receptive_field_analysis.receptive_field as rf
import matplotlib.pyplot as plt


###############################################################################
# Next, we will set some parameters

# these determine which cell and experiment will be analyzed
cell_specimen_id = 587377366
ophys_experiment_id = 501474098

# this specifies a filesystem location where downloaded data will be stored
manifest_file = '../boc/manifest.json'


###############################################################################
# In order to access the data, we will use the BrainObservatoryCache class. 
# This class downloads data on demand and caches it locally to prevent re-downloading.

boc = BrainObservatoryCache( manifest_file=manifest_file )

data_set = boc.get_ophys_experiment_data( ophys_experiment_id )
cell_index = data_set.get_cell_specimen_indices([ cell_specimen_id ])[0]


###############################################################################
# With the data in memory, we can carry out the analyses.

rf_data = rf.compute_receptive_field_with_postprocessing(data_set, 
                                                         cell_index, 
                                                         'locally_sparse_noise', 
                                                         alpha=0.5, 
                                                         number_of_shuffles=10000)


###############################################################################
# A :math:`\chi^2` test is applied to each pixel in order to determine if
# cellular responses to stimuli in the neighboorhood of that pixel are non-uniformly 
# distributed. The results are represented here as a heatmap of negative log-likelihoods
# for the null hypothesis (higher values mean that a receptive field in the neighborhood of the 
# pixel is more likely).

rfvis.plot_chi_square_summary(rf_data)

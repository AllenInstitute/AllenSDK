r'''

Comparing Stimulus Blocks
=========================

Newer experiments switched from using a single locally sparse noise stimulus with 
4.54 visual-degree pixels to two blocks of stimuli with different pixel sizes 
(a 4.65° block and an 9.3° block that are each half the 
length of the original 4.65°-only stimulus). 
You can characterize the receptive fields from reponses to each stimulus block separately.
Here we'll show you how to compare data within a cell across the two blocks.

'''


###############################################################################
# imports 


from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.receptive_field_analysis.visualization as rfvis
import allensdk.brain_observatory.receptive_field_analysis.receptive_field as rf
import matplotlib.pyplot as plt


###############################################################################
# Choose a cell and experiment.


cell_specimen_id = 559109414
ophys_experiment_id = 558599066

manifest_file = '../boc/manifest.json'


###############################################################################
# Load up the data.


boc = BrainObservatoryCache( manifest_file=manifest_file )

data_set = boc.get_ophys_experiment_data( ophys_experiment_id )
cell_index = data_set.get_cell_specimen_indices([ cell_specimen_id ])[0]


###############################################################################
# We'll compute the receptive field analyses for the low and high resolution blocks.
# 
# Careful here:  the NWB files refer to these stimuli as locally_sparse_noise_8deg (9.3°)
# and locally_sparse_noise_4deg (4.65°) respectively. 


rf_data_low_res = rf.compute_receptive_field_with_postprocessing(
    data_set, cell_index, 'locally_sparse_noise_8deg', alpha=0.5, number_of_shuffles=10000)


rf_data_high_res = rf.compute_receptive_field_with_postprocessing(
    data_set, cell_index, 'locally_sparse_noise_4deg', alpha=0.5, number_of_shuffles=10000)


###############################################################################
# Now we plot the (blurred) response-triggered stimulus field for the high-resolution block ...


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 3))
rfvis.plot_rts_blur_summary(rf_data_high_res, ax1, ax2)


###############################################################################
# ... and for the low-resolution block


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 3))
rfvis.plot_rts_blur_summary(rf_data_low_res, ax1, ax2)

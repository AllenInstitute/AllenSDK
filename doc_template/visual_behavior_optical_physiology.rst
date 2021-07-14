Visual Behavior - Optical Physiology
====================================

The Visual Behavior 2P project used in vivo 2-photon calcium imaging (also 
called optical physiology, or “ophys”) to measure the activity of populations 
of genetically identified neurons in the visual cortex of mice performing a 
visually guided behavioral task (image change detection, described below). We used single- and 
multi-plane imaging approaches to record the activity of populations of 
excitatory neurons and two inhibitory classes, Somatostatin (Sst) and 
Vasoactive Intestinal Peptide (Vip) expressing interneurons, across 
multiple cortical depths and two visual areas (VSIp and VISl). Each population of neurons was 
imaged repeatedly over multiple days under different sensory and behavioral 
contexts, including with familiar and novel stimuli, as well as active behavior 
and passive viewing conditions. This dataset can be used to evaluate the 
influence of experience, expectation, and task engagement on neural coding 
and dynamics.  

.. image:: /_static/visual_behavior_2p/datasets.png
   :align: center
   :width: 850

While 2-photon imaging data was acquired in well-trained mice, the full 
behavioral training history of all imaged mice is also provided, allowing 
investigation into task learning, behavioral strategy, and inter-animal 
variability.

Overall, the dataset includes neural and behavioral measurements from 82 
mice, including 3021 behavior training sessions and 551 in vivo 2-photon 
imaging sessions, resulting in longitudinal recordings from 34,619 
cortical cells. 

The table below describes the numbers of mice, sessions, and unique recorded 
neurons for each transgenic line and imaging platform in the dataset:

.. image:: /_static/visual_behavior_2p/variants_table.png
   :align: center
   :width: 850

GETTING STARTED
---------------

First, `install or update the AllenSDK <https://allensdk.readthedocs.io/en/latest/install.html>`_, 
our Python based toolkit for accessing and working with Allen Institute datasets.

Data is provided in in `NWB <https://www.nwb.org/>`_ format and can be downloaded using the AllenSDK, 
or accessed directly via an S3 bucket (instructions provided in notebook #1 below). Regardless of which method of file 
download you choose, we recommend that you load and interact with the data 
using the tools provided in the AllenSDK, which have been designed to simplify 
data access and subsequent analysis. No knowledge of the NWB file format is required.

Specific information about how Visual Behavior Optical Physiology data is stored 
in NWB files and how AllenSDK accesses NWB files can be found `here <visual_behavior_ophys_nwb.html>`_.

To get started, check out these jupyter notebooks to learn how to:

1) `Download data using the AllenSDK or directly from our Amazon S3 bucket <_static/examples/nb/visual_behavior_ophys_data_access.html>`_ `(download .ipynb) <_static/examples/nb/visual_behavior_ophys_data_access.ipynb>`_ `(Open in Colab) <https://colab.research.google.com/github/AllenInstitute/allenSDK/blob/master/doc_template/examples_root/examples/nb/visual_behavior_ophys_data_access.ipynb>`_
2) `Identify experiments of interest using the dataset manifest <_static/examples/nb/visual_behavior_ophys_dataset_manifest.html>`_ `(download .ipynb) <_static/examples/nb/visual_behavior_ophys_dataset_manifest.ipynb>`_ `(Open in Colab) <https://colab.research.google.com/github/AllenInstitute/allenSDK/blob/master/doc_template/examples_root/examples/nb/visual_behavior_ophys_dataset_manifest.ipynb>`_
3) `Load and visualize data from a 2-photon imaging experiment <_static/examples/nb/visual_behavior_load_ophys_data.html>`_ `(download .ipynb) <_static/examples/nb/visual_behavior_load_ophys_data.ipynb>`_ `(Open in Colab) <https://colab.research.google.com/github/AllenInstitute/allenSDK/blob/master/doc_template/examples_root/examples/nb/visual_behavior_load_ophys_data.ipynb>`_
4) `Examine the full training history of one mouse <_static/examples/nb/visual_behavior_mouse_history.html>`_ `(download .ipynb) <_static/examples/nb/visual_behavior_mouse_history.ipynb>`_ `(Open in Colab) <https://colab.research.google.com/github/AllenInstitute/allenSDK/blob/master/doc_template/examples_root/examples/nb/visual_behavior_mouse_history.ipynb>`_
5) `Compare behavior and neural activity across different trial types in the task <_static/examples/nb/visual_behavior_compare_across_trial_types.html>`_ `(download .ipynb) <_static/examples/nb/visual_behavior_compare_across_trial_types.ipynb>`_ `(Open in Colab) <https://colab.research.google.com/github/AllenInstitute/allenSDK/blob/master/doc_template/examples_root/examples/nb/visual_behavior_compare_across_trial_types.ipynb>`_



For a description of available AllenSDK methods and attributes for data access, see this 
`further documentation <https://alleninstitute.sharepoint.com/:w:/s/VisualBehaviorAIBS/EUkWXB9X8wZKleIGtsviscMBTgesWXsrHESs84Ye9FvqzQ?e=Jm7GmA>`_.

For detailed information about the experimental design, data acquisition, 
and informatics methods, please refer to our `technical whitepaper <https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/4e/be/4ebe2911-bd38-4230-86c8-01a86cfd758e/visual_behavior_2p_technical_whitepaper.pdf>`_.

If you have questions about the dataset that aren’t addressed by the whitepaper 
or any of our tutorials, please reach out by posting at 
https://community.brain-map.org/  

CHANGE DETECTION TASK
---------------------

.. image:: /_static/visual_behavior_2p/task.png
   :align: center
   :width: 850

We trained mice to perform a go/no-go visual change detection task in 
which they learned to lick a spout in response to changes in stimulus 
identity to earn a water reward. Visual stimuli are continuously presented 
over a 1-hour session, with no explicit cue to indicate the start of a 
trial. Mice are free to run on a circular disk during the session.

We used a standardized procedure to progress mice through a series of 
training stages, with transitions between stages determined by specific 
advancement criteria. First, mice learned to detect changes in the 
orientation of full field static grating stimuli. Next, a 500ms inter 
stimulus interval period with mean luminance gray screen was added between 
the 250ms stimulus presentations, incorporating a short-term memory component 
to the task. Once mice successfully and consistently performed the orientation 
change detection with flashed gratings, they moved to the image change 
detection version of the task. During image change detection, 8 natural scene 
images were presented during each behavioral session, for a total of 64 
possible image transitions. When behavioral performance again reached 
criterion, mice were transitioned to the 2-photon imaging stage in which they 
performed the task under a microscope to allow simultaneous measurement of 
neural activity and behavior. 

.. image:: /_static/visual_behavior_2p/automated_training.png
   :align: center
   :width: 850

Behavioral training data for mice progressing through these 
stages of task learning is accessible via the **BehaviorSession** 
class of the AllenSDK or the :py:meth:`.get_behavior_session()` method of 
the **VisualBehaviorOphysProjectCache**. Each **BehaviorSession** 
contains the following data streams, event times, and metadata:

- Running speed
- Lick times
- Reward times
- Stimulus presentations
- Behavioral trial information
- Mouse metadata (age, sex, genotype, etc) 

.. image:: /_static/visual_behavior_2p/behavior_timeseries.png
   :align: center
   :width: 850

2-PHOTON IMAGING DATASET
------------------------

Once mice are well-trained on the image change detection task, 
they transition to performing the behavior under a 2-photon 
microscope. During the imaging phase, mice undergo multiple 
**session types**, allowing measurement of neural activity across 
different sensory and behavioral contexts. 

.. image:: /_static/visual_behavior_2p/experiment_design.png
   :align: center
   :width: 850

Mice initially perform the task under the microscope with the same set of 
images they observed during training, which have become highly familiar 
(each image is viewed thousands of times during training). Mice also 
undergo several sessions with a novel image set that they had not seen 
prior to the 2-photon imaging portion of the experiment. Interleaved 
between active behavior sessions, are passive viewing sessions where the 
mice are given their daily water before the session (and are thus satiated) 
and view the task stimuli with the lick spout retracted so they are unable 
to earn water rewards. This allows investigation of the impact of motivation 
and attention on patterns of neural activity. Finally, stimuli were randomly 
omitted with a 5% probability, resulting in an extended gray screen period 
between two presentations of the same stimulus, and disrupting the expected 
cadence of stimulus presentations. Stimuli were only omitted during the 
2-photon imaging sessions (not during training), and change stimuli were 
never omitted.

We used both single- and multi-plane 2-photon imaging to record the activity 
of GCaMP6 expressing cells in populations of excitatory 
(Slc17a7-IRES2-Cre;Camk2a-tTA;Ai93(TITL-GCaMP6)) and inhibitory 
(Vip-IRES-Cre;Ai148(TIT2L-GC6f-ICL-tTA2) & Sst-IRES-Cre;Ai148(TIT2L-GC6f-ICL-tTA2)) 
neurons. Imaging took place between 75-400um below the cortical surface. 

.. image:: /_static/visual_behavior_2p/cre_lines.png
   :align: center
   :width: 850

The data collected in a single continuous recording is defined as a 
**session**. For single-plane imaging experiments, there is only one 
imaging plane (referred to as an **experiment**) per session. For 
multi-plane imaging experiments, there can be up to 8 imaging planes 
(aka 8 experiments) per session. Due to our strict QC process, described 
below, not all multi-plane imaging sessions have exactly 8 experiments, 
as some imaging planes did not meet our data quality criteria. 

We aimed to track the activity of single neurons across the session 
types described above by targeting the same population of neurons over 
multiple recording sessions, with only one session recorded per day 
for a given mouse. The collection of imaging sessions for a given 
population of cells, belonging to a single imaging plane measured 
across days, is called a **container**. A container can include between 
3 and 11 separate sessions for a given imaging plane. Mice imaged 
with the multi-plane 2-photon microscope can have multiple containers, 
one for each imaging plane recorded across multiple sessions. The session 
types available for a given container can vary, due to our selection 
criteria to ensure data quality (described below).

Thus, each mouse can have one or more **containers**, each representing a 
unique imaging plane (**experiment**) that has been targeted across 
multiple recording **sessions**, under different behavioral and 
sensory conditions (**session types**).

.. image:: /_static/visual_behavior_2p/data_structure.png
   :align: center
   :width: 850

The **BehaviorOphysExperiment** class in the AllenSDK (or the 
:py:meth:`.get_behavior_ophys_experiment()` method of the 
**VisualBehaviorOphysProjectCache**) provides all data for a 
single imaging plane, recorded in a single session, and contains 
the following data streams in addition to the behavioral data 
described above:

- Max intensity image
- Average intensity image
- Segmentation masks
- dF/F traces (baseline corrected, normalized fluorescence traces)
- Corrected fluorescence traces (neuropil subtracted and demixed, but not normalized)
- Events (detected with an L0 event detection algorithm)
- Pupil position
- Pupil area

.. image:: /_static/visual_behavior_2p/behavior_and_ophys_timeseries.png
   :align: center
   :width: 850

DATA PROCESSING
---------------

Each 2-photon movie is processed through a series of steps to 
obtain single cell traces of baseline-corrected fluorescence (dF/F) 
and extracted events, that are packaged into NWB files along with 
stimulus and behavioral information, as well as other metadata. 

.. image:: /_static/visual_behavior_2p/data_processing.png
   :align: center
   :width: 850

Detailed descriptions of data processing steps can be found 
in the technical white paper, as well as our 
`data processing repository <https://github.com/AllenInstitute/ophys_etl_pipelines>`_.


QUALITY CONTROL
---------------

Every 2-photon imaging session was carefully evaluated for a variety 
of quality control criteria to ensure that the final dataset is of 
the highest quality possible. Sessions or imaging planes that do not 
meet our criteria are excluded from the dataset in this release. These 
are a few of the key aspects of the data that are evaluated:

- intensity drift
- image saturation or bleaching
- z-drift over the course of a session
- accuracy of session-to-session field of view matching
- excessive or uncorrectable motion in the image
- uncorrectable crosstalk between simultaneously recorded multiscope planes
- errors affecting temporal alignment of data streams
- hardware or software failures
- brain health
- animal stress

SUMMARY OF AVAILABLE DATA
-------------------------

.. list-table:: 
   :widths: 50 50 50
   :header-rows: 1

   * - Behavior
     - Physiology
     - Metadata
   * - Running speed
     - Max intensity projection image
     - Mouse genotype, age, sex 
   * - Licks
     - Average projection image
     - Date of acquisition
   * - Rewards
     - Segmentation mask image
     - Imaging parameters
   * - Pupil area
     - Cell specimen table
     - Task parameters
   * - Pupil position
     - Cell ROI masks
     - Session type
   * - Stimulus presentations table
     - Corrected fluorescence traces
     - Stimulus images
   * - Trials table
     - dF/F activity traces
     - Performance metrics
   * - Stimulus timestamps
     - Detected events
     - 
   * - 
     - Ophys timestamps
     - 

DATA FILE CHANGELOG
-------------------

**v0.3.0**

13 sessions were labeled with the wrong session_type in v0.2.0. We have
corrected that error. The offending sessions were

.. list-table:: 
   :widths: 30 30 50 50
   :header-rows: 1

   * - behavior_session_id
     - ophys_session_id
     - session_type_v0.2.0
     - session_type_v0.3.0
   * - 875020233
     -
     - OPHYS_3_images_A
     - OPHYS_2_images_A_passive
   * - 902810506
     -
     - TRAINING_4_images_B_training
     - TRAINING_3_images_B_10uL_reward
   * - 914219174
     -
     - OPHYS_0_images_B_habituation
     - TRAINING_5_images_B_handoff_ready
   * - 863571063
     -
     -  TRAINING_5_images_A_handoff_ready
     - TRAINING_1_gratings
   * - 974330793
     -
     - OPHYS_0_images_B_habituation
     - TRAINING_5_images_B_handoff_ready
   * - 863571072
     -
     - OPHYS_5_images_B_passive
     - TRAINING_4_images_A_training
   * - 1010972317
     -
     - OPHYS_4_images_A
     - OPHYS_3_images_B
   * - 1011659817
     -
     - OPHYS_5_images_A_passive
     - OPHYS_4_images_A
   * - 1003302686
     - 1003277121
     - OPHYS_6_images_A
     - OPHYS_5_images_A_passive
   * - 863571054
     -
     - OPHYS_7_receptive_field_mapping
     - TRAINING_5_images_A_epilogue
   * - 974282914
     - 974167263
     - OPHYS_6_images_B
     - OPHYS_5_images_B_passive
   * - 885418521
     -
     - OPHYS_1_images_A
     - TRAINING_5_images_A_handoff_lapsed
   * - 915739774
     -
     - OPHYS_1_images_A
     - OPHYS_0_images_A_habituation

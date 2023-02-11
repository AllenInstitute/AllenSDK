GETTING STARTED
---------------

Prerequisites
^^^^^^^^^
* `install or update the AllenSDK <https://allensdk.readthedocs.io/en/latest/install.html>`_,
    our Python based toolkit for accessing and working with Allen Institute datasets.
* `Pandas <https://pandas.pydata.org/docs/user_guide/index.html>`_ familiarity

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
`further documentation <https://visual-behavior-ophys-data.s3.us-west-2.amazonaws.com/visual-behavior-ophys/VBP_WhitePaper_SDK_Documentation.pdf>`_.

For detailed information about the experimental design, data acquisition, 
and informatics methods, please refer to our `technical whitepaper <https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/4e/be/4ebe2911-bd38-4230-86c8-01a86cfd758e/visual_behavior_2p_technical_whitepaper.pdf>`_.

If you have questions about the dataset that aren’t addressed by the whitepaper 
or any of our tutorials, please reach out by posting at 
https://community.brain-map.org/  

VISUAL BEHAVIOR OPTICAL PHYSIOLOGY DATASETS
--------------------------------------------

The Visual Behavior 2P project used in vivo 2-photon calcium imaging (also 
called optical physiology, or “ophys”) to measure the activity 
of genetically identified neurons in the visual cortex of mice performing a 
go/no-go visual change detection task. This dataset can be used to evaluate the 
influence of experience, expectation, and task engagement on neural coding 
and dynamics in excitatory and inhibitory cell populations.  

.. image:: /_static/visual_behavior_2p/visual_behavior.png
   :align: center
   :width: 850

We used single- and multi-plane imaging 
approaches to record the activity of populations of neurons across multiple 
cortical depths and visual areas. Each population of neurons was imaged 
repeatedly over multiple days under different sensory and behavioral 
contexts, including familiar and novel stimuli, as well as active behavior 
and passive viewing conditions. 

.. image:: /_static/visual_behavior_2p/experimental_design_session_types.png
   :align: center
   :width: 850

Different imaging configurations and stimulus sets were used in 
different groups of mice, resulting in four unique datasets (indicated by their "project code"). 
Two single-plane 2-photon datasets were acquired in the primary visual cortex (VISp),
and the image set that became familiar over training was swapped to control for novelty effects.
One multi-plane dataset was acquired at 4 cortical depths in 2 visual areas (VISp & VISl).
Another multi-plane dataset was acquired at 2 cortical depths in 4 visual areas (VISp, VISl, VISal, VISam).
In this dataset, two of the images that became highly familiar during training ("shared images") were interleaved
among novel images during the novel sessions.

.. image:: /_static/visual_behavior_2p/dataset_variants_GH.png
   :align: center
   :width: 850

For each dataset, we imaged the activity of GCaMP6 expressing cells in populations 
of excitatory (Slc17a7-IRES2-Cre;Camk2a-tTA;Ai93(TITL-GCaMP6f) or Ai94(TITL-GCaMP6s)), 
Vip inhibitory (Vip-IRES-Cre;Ai148(TIT2L-GCaMP6f-ICL-tTA2)), 
and Sst inhibitory (Sst-IRES-Cre;Ai148(TIT2L-GCaMP6f-ICL-tTA2)) neurons. 
Imaging took place between 75-400um below the cortical surface. 

.. image:: /_static/visual_behavior_2p/cre_lines.png
   :align: center
   :width: 850

Overall, the dataset includes neural and behavioral measurements from 107 
mice during 704 in vivo 2-photon imaging sessions from 326 unique fields of view, 
resulting in longitudinal recordings from 50,482 cortical neurons. 

The table below describes the numbers of mice, sessions, and unique recorded 
neurons for each transgenic line and experimental configuration (project code):

.. image:: /_static/visual_behavior_2p/final_dataset_numbers.png
   :align: center
   :width: 850


BEHAVIORAL TRAINING
---------------------

Prior to 2-photon imaging, mice were trained to perform a go/no-go visual 
change detection task in which they learned to lick a spout in response to 
changes in stimulus identity to earn a water reward. The full 
behavioral training history of all imaged mice is available (4,787 behavior sessions), 
allowing investigation into task learning, behavioral strategy, and inter-animal 
variability.

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
criterion (d-prime >1 for 2 out of 3 consecutive days), 
mice were transitioned to the 2-photon imaging stage in which they 
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


2-PHOTON IMAGING DURING BEHAVIOR
---------------------------------

Once mice are well-trained on the image change detection task, 
they transition to performing the behavior under a 2-photon 
microscope. During the imaging phase, mice undergo multiple 
**session types**, allowing measurement of neural activity across 
different sensory and behavioral contexts. 

.. image:: /_static/visual_behavior_2p/expt_design_notes.png
   :align: center
   :width: 850

Mice initially perform the task under the microscope with the same set of 
images they observed during training, which have become highly familiar 
(each image is viewed thousands of times during training). Mice also 
undergo several sessions with a novel image set that they had not seen 
prior to the 2-photon imaging portion of the experiment. Passive viewing 
sessions are interleaved between active behavior sessions.
On passive days, mice are given their daily water before the session (and are thus satiated) 
and view the task stimuli in open loop mode, with the lick spout retracted 
to indicate that rewards are not availabl. This allows investigation of the 
impact of motivation and attention on patterns of neural activity. 

During ophys sessions (but not during training), stimuli were randomly 
omitted with a 5% probability, resulting in an extended gray screen period 
between two presentations of the same stimulus, and disrupting the expected 
cadence of stimulus presentations. The change and pre-change stimulus presentations 
were never omitted. 

.. image:: /_static/visual_behavior_2p/data_streams.png
   :align: center
   :width: 850

The **BehaviorOphysExperiment** class in the AllenSDK (or the 
:py:meth:`.get_behavior_ophys_experiment()` method of the 
**VisualBehaviorOphysProjectCache**) provides all data for a 
single imaging plane, recorded in a single session, and contains 
the following data streams:

- Maximum intensity image
- Average intensity image
- Segmentation masks and ROI metadata
- dF/F traces (baseline corrected, normalized fluorescence traces)
- Corrected fluorescence traces (neuropil subtracted and demixed, but not normalized)
- Events (detected with an L0 event detection algorithm)
- Pupil position
- Pupil area
- Running speed
- Lick times
- Reward times
- Stimulus presentations
- Behavioral trial information
- Mouse metadata (age, sex, genotype, etc) 

.. image:: /_static/visual_behavior_2p/data_structure.png
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
3 and 11 separate sessions for that imaging plane. Mice imaged 
with the multi-plane 2-photon microscope can have multiple containers, 
one for each imaging plane recorded across multiple sessions. The session 
types available for a given container can vary, due to our selection 
criteria to ensure data quality (described below).

Thus, each mouse can have one or more **containers**, each representing a 
unique imaging plane (**experiment**) that has been targeted on 
multiple recording days (**sessions**), under different behavioral and 
sensory conditions (**session types**).


DATA PROCESSING
---------------

Each 2-photon movie is processed through a series of steps to 
obtain single cell traces of baseline-corrected fluorescence (dF/F) 
and detected events, and packaged into the NWB file format along with 
stimulus and behavioral information, as well as other metadata. 

Detailed descriptions of data processing steps can be found 
in the technical white paper, as well as our 
`data processing repository <https://github.com/AllenInstitute/ophys_etl_pipelines>`_.

.. image:: /_static/visual_behavior_2p/data_processing.png
   :align: center
   :width: 850


QUALITY CONTROL
---------------

Every 2-photon imaging session was carefully evaluated for a variety 
of quality control criteria to ensure that the final dataset is of 
the highest quality possible. Sessions or imaging planes that do not 
meet our criteria are excluded from the released dataset. These 
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

**v1.0.1**

Metadata corrections
- ophys_container_id columns contained extra IDs of incorrect containers.

**v1.0.0**

New Data

- 107 mice, up from 82
- 4082 behavior training sessions, up from 3021.
- 705 in vivo 2-photon imaging sessions, up from 551.
- 50,489 logitudinal recordings from cortical cells, up from 34,619

Metadata changes

- A new metadata table is present: ophys_cells_table. This table has a project-wide aggregate of cell_specimen_id, cell_roi_id, and ophys_experiment_id.
- Added 'experience_level', 'passive' and 'image_set' columns to ophys_experiments_table

Data Corrections

- 196 BehaviorOphysExperiments had excess invalid ROIs in the dataframe returned by
  the events field. These have been corrected to remove these invalid ROIs.

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

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

1) `Download data using the AllenSDK or directly from our Amazon S3 bucket <_static/examples/nb/visual_behavior_neuropixels_data_access.html>`_ `(download .ipynb) <_static/examples/nb/visual_behavior_neuropixels_data_access.ipynb>`_
2) `Plot quality metrics for the 'units' identified in these experiments <_static/examples/nb/visual_behavior_neuropixels_quality_metrics.html>`_ `(download .ipynb) <_static/examples/nb/visual_behavior_neuropixels_quality_metrics.ipynb>`_


If you have questions about the dataset that aren’t addressed by the whitepaper
or any of our tutorials, please reach out by posting at
https://community.brain-map.org/

Visual Behavior - Neuropixels
====================================

**Needs content**

CHANGE DETECTION TASK
---------------------

**Copied from VBO content**

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


Neuropixels DATASET
-------------------

**Needs content***


DATA PROCESSING
---------------

**Needs content**


SUMMARY OF AVAILABLE DATA
-------------------------

**Needs content**

DATA FILE CHANGELOG
-------------------

**Needs content**

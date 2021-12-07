import os
import numpy as np
from typing import Optional, List

import imageio
from pynwb import NWBFile

from allensdk.brain_observatory import nwb
from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    StimulusFileReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.stimuli.presentations \
    import \
    Presentations
from allensdk.brain_observatory.behavior.stimulus_processing import \
    get_stimulus_templates
from allensdk.brain_observatory.behavior.data_objects.stimuli \
    .stimulus_templates import \
    StimulusTemplate, StimulusTemplateFactory
from allensdk.brain_observatory.behavior.write_nwb.extensions\
    .stimulus_template.ndx_stimulus_template import \
    StimulusTemplateExtension
from allensdk.internal.core.lims_utilities import safe_system_path

from allensdk.brain_observatory.behavior.data_objects.stimuli.densemovie_presentations import get_original_stim_name, stim_name_parse
from pathlib import Path


class DenseMovieTemplates(DataObject, StimulusFileReadableInterface,
                NwbReadableInterface, NwbWritableInterface):
    def __init__(self, templates:  dict):
        super().__init__(name='stimulus_templates', value=templates)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: StimulusFile) -> "DenseMovieTemplates":
        """Get stimulus templates (movies, scenes) for behavior session.
        
        FOR NOW:  This returns a dict of dicts for warped and unwarped stimuli.  Keys are stim names from the presentation table and values are paths to the npy array."""

        warped_path = Path('/allen/programs/braintv/workgroups/nc-ophys/ImageData/Dan/ten_session_movies')
        unwarped_path = Path('/allen/programs/braintv/workgroups/cortexmodels/michaelbu/Stimuli/SignalNoise/arrays')

        stim_dict = {'warped': {}, 'unwarped': {}}

        pkl_data = stimulus_file.data

        for stim in pkl_data['stimuli']:

            warped_stim_name = str(stim['movie_path']).split('\\')[-1]

            stage_number, segment_number, test_or_train = stim_name_parse(warped_stim_name)
            original_stim_name = get_original_stim_name(stage_number, segment_number, test_or_train)

            stim_dict['warped'][warped_stim_name] = warped_path / warped_stim_name
            stim_dict['unwarped'][original_stim_name] = unwarped_path / original_stim_name

        return DenseMovieTemplates(templates=stim_dict)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "DenseMovieTemplates":
        raise NotImplementedError

    def to_nwb(self, nwbfile: NWBFile,
               stimulus_presentations: Presentations) -> NWBFile:
        raise NotImplementedError
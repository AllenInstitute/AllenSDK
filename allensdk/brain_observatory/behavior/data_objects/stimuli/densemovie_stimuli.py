from typing import Optional, List

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    StimulusFileReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.stimuli.densemovie_presentations \
    import \
    DenseMoviePresentations
from allensdk.brain_observatory.behavior.data_objects.stimuli.densemovie_templates \
    import \
    DenseMovieTemplates


class DenseMovieStimuli(DataObject, StimulusFileReadableInterface,
              NwbReadableInterface, NwbWritableInterface):
    def __init__(self, presentations: DenseMoviePresentations,
                 templates: DenseMovieTemplates):
        super().__init__(name='stimuli', value=self)
        self._presentations = presentations
        self._templates = templates

    @property
    def presentations(self) -> DenseMoviePresentations:
        return self._presentations

    @property
    def templates(self) -> DenseMovieTemplates:
        return self._templates

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "DenseMovieStimuli":
        p = DenseMoviePresentations.from_nwb(nwbfile=nwbfile)
        t = DenseMovieTemplates.from_nwb(nwbfile=nwbfile)
        return DenseMovieStimuli(presentations=p, templates=t)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: StimulusFile,
            stimulus_timestamps: StimulusTimestamps) -> "DenseMovieStimuli":
        p = DenseMoviePresentations.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps)
        t = DenseMovieTemplates.from_stimulus_file(stimulus_file=stimulus_file)
        return DenseMovieStimuli(presentations=p, templates=t)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        nwbfile = self._templates.to_nwb(
            nwbfile=nwbfile, stimulus_presentations=self._presentations)
        nwbfile = self._presentations.to_nwb(nwbfile=nwbfile)

        return nwbfile

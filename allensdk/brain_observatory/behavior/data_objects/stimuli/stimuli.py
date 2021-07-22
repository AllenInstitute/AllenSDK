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
from allensdk.brain_observatory.behavior.data_objects.stimuli.presentations \
    import \
    Presentations
from allensdk.brain_observatory.behavior.data_objects.stimuli.templates \
    import \
    Templates


class Stimuli(DataObject, StimulusFileReadableInterface,
              NwbReadableInterface, NwbWritableInterface):
    def __init__(self, presentations: Presentations,
                 templates: Templates):
        super().__init__(name='stimuli', value=self)
        self._presentations = presentations
        self._templates = templates

    @property
    def presentations(self) -> Presentations:
        return self._presentations

    @property
    def templates(self) -> Templates:
        return self._templates

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "Stimuli":
        p = Presentations.from_nwb(nwbfile=nwbfile)
        t = Templates.from_nwb(nwbfile=nwbfile)
        return Stimuli(presentations=p, templates=t)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: StimulusFile,
            stimulus_timestamps: StimulusTimestamps,
            limit_to_images: Optional[List] = None) -> "Stimuli":
        p = Presentations.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            limit_to_images=limit_to_images)
        t = Templates.from_stimulus_file(stimulus_file=stimulus_file,
                                         limit_to_images=limit_to_images)
        return Stimuli(presentations=p, templates=t)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        nwbfile = self._templates.to_nwb(
            nwbfile=nwbfile, stimulus_presentations=self._presentations)
        nwbfile = self._presentations.to_nwb(nwbfile=nwbfile)

        return nwbfile

from typing import Optional, List

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.core import DataObject
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.core import \
    NwbReadableInterface
from allensdk.brain_observatory.behavior.data_files.stimulus_file import \
    StimulusFileReadableInterface
from allensdk.core import NwbWritableInterface
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
        super().__init__(name='stimuli', value=None, is_value_self=True)
        self._presentations = presentations
        self._templates = templates

    @property
    def presentations(self) -> Presentations:
        return self._presentations

    @property
    def templates(self) -> Templates:
        return self._templates

    @classmethod
    def from_nwb(
            cls,
            nwbfile: NWBFile,
            presentation_columns: Optional[List[str]] = None,
            add_is_change_to_presentations_table=True
    ) -> "Stimuli":
        p = Presentations.from_nwb(
            nwbfile=nwbfile,
            column_list=presentation_columns,
            add_is_change=add_is_change_to_presentations_table)
        t = Templates.from_nwb(nwbfile=nwbfile)
        return Stimuli(presentations=p, templates=t)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: BehaviorStimulusFile,
            stimulus_timestamps: StimulusTimestamps,
            behavior_session_id: int,
            limit_to_images: Optional[List] = None,
            presentation_columns: Optional[List[str]] = None,
            presentation_fill_omitted_values: bool = True
    ) -> "Stimuli":
        """

        Parameters
        ----------
        stimulus_file
        stimulus_timestamps
        behavior_session_id
            behavior session id in LIMS
        limit_to_images: limit to certain images. Used for testing.
        presentation_columns: The columns and order of columns
            in the final presentations dataframe
        presentation_fill_omitted_values: Whether to fill stop_time and
            duration for omitted frames

        Returns
        -------

        """
        p = Presentations.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            behavior_session_id=behavior_session_id,
            limit_to_images=limit_to_images,
            column_list=presentation_columns,
            fill_omitted_values=presentation_fill_omitted_values
        )
        t = Templates.from_stimulus_file(stimulus_file=stimulus_file,
                                         limit_to_images=limit_to_images)
        return Stimuli(presentations=p, templates=t)

    def to_nwb(self, nwbfile: NWBFile,
               presentations_stimulus_column_name='stimulus_name') -> NWBFile:
        """

        Parameters
        ----------
        nwbfile
        presentations_stimulus_column_name: Name of the column in the
            presentations table that denotes the stimulus name

        Returns
        -------
        NWBFile
        """
        nwbfile = self._templates.to_nwb(
            nwbfile=nwbfile, stimulus_presentations=self._presentations)
        nwbfile = self._presentations.to_nwb(
            nwbfile=nwbfile,
            stimulus_name_column=presentations_stimulus_column_name)

        return nwbfile

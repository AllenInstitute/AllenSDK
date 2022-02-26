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


class EcephysStimuli(DataObject, StimulusFileReadableInterface,
              NwbReadableInterface, NwbWritableInterface):
    def __init__(self, presentations: Presentations,
                 templates: Templates):
        super().__init__(name='stimuli', value=self)
        self._presentations = presentations
        self._templates = templates

        read_stimulus_table(stimulus_table_path, columns_to_drop=stimulus_columns_to_drop)

    
    def read_stimulus_table(path: str,
                            column_renames_map: Dict[str, str] = None,
                            columns_to_drop: List[str] = None) -> pd.DataFrame:
        """ Loads from a CSV on disk the stimulus table for this session.
        Optionally renames columns to match NWB epoch specifications.

        Parameters
        ----------
        path : str
            path to stimulus table csv
        column_renames_map : Dict[str, str], optional
            If provided, will be used to rename columns from keys -> values.
            Default renames: ('Start' -> 'start_time') and ('End' -> 'stop_time')
        columns_to_drop : List, optional
            A list of column names to drop. Columns will be dropped BEFORE
            any renaming occurs. If None, no columns are dropped.
            By default None.

        Returns
        -------
        pd.DataFrame :
            stimulus table with applied renames

        """
        if column_renames_map is None:
            column_renames_map = STIM_TABLE_RENAMES_MAP

        ext = PurePath(path).suffix

        if ext == ".csv":
            stimulus_table = pd.read_csv(path)
        else:
            raise IOError(f"unrecognized stimulus table extension: {ext}")

        if columns_to_drop:
            stimulus_table = stimulus_table.drop(errors='ignore',
                                                 columns=columns_to_drop)

        return stimulus_table.rename(columns=column_renames_map, index={})

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
    def from_stimulus_table(cls, stimulus_table: EcephysStimulusTable) -> "EcephysStimuli":
        stimulus_columns_to_drop = [
            "colorSpace", "depth", "interpolate", "pos", "rgbPedestal", "tex",
            "texRes", "flipHoriz", "flipVert", "rgb", "signalDots"
        ]

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

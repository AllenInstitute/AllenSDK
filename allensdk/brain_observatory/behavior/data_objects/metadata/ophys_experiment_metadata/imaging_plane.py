from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import SyncFile
from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    InternalReadableInterface, JsonReadableInterface, NwbReadableInterface
from allensdk.internal.api import PostgresQueryMixin


class ImagingPlane(DataObject, InternalReadableInterface,
                   JsonReadableInterface, NwbReadableInterface):
    def __init__(self, ophys_frame_rate: float,
                 targeted_structure: str,
                 excitation_lambda: float):
        super().__init__(name='imaging_plane', value=self)
        self._ophys_frame_rate = ophys_frame_rate
        self._targeted_structure = targeted_structure
        self._excitation_lambda = excitation_lambda

    @classmethod
    def from_internal(cls, ophys_experiment_id: int,
                      lims_db: PostgresQueryMixin,
                      excitation_lambda=910.0) -> "ImagingPlane":
        sync_file = SyncFile.from_lims(ophys_experiment_id=ophys_experiment_id,
                                       db=lims_db)
        ophys_frame_rate = cls._get_frame_rate_from_sync_file(
            sync_file=sync_file)
        targeted_structure = cls._get_targeted_structure_from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        return cls(ophys_frame_rate=ophys_frame_rate,
                   targeted_structure=targeted_structure,
                   excitation_lambda=excitation_lambda)

    @classmethod
    def from_json(cls, dict_repr: dict,
                  excitation_lambda=910.0) -> "ImagingPlane":
        targeted_structure = dict_repr['targeted_structure']
        sync_file = SyncFile.from_json(dict_repr=dict_repr)
        ophys_fame_rate = cls._get_frame_rate_from_sync_file(
            sync_file=sync_file)
        return cls(targeted_structure=targeted_structure,
                   ophys_frame_rate=ophys_fame_rate,
                   excitation_lambda=excitation_lambda)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "ImagingPlane":
        ophys_module = nwbfile.processing['ophys']
        image_seg = ophys_module.data_interfaces['image_segmentation']
        imaging_plane = image_seg.plane_segmentations[
            'cell_specimen_table'].imaging_plane
        ophys_frame_rate = imaging_plane.imaging_rate
        targeted_structure = imaging_plane.location
        excitation_lambda = imaging_plane.excitation_lambda
        return cls(ophys_frame_rate=ophys_frame_rate,
                   targeted_structure=targeted_structure,
                   excitation_lambda=excitation_lambda)

    @property
    def ophys_frame_rate(self) -> float:
        return self._ophys_frame_rate

    @property
    def targeted_structure(self) -> str:
        return self._targeted_structure

    @property
    def excitation_lambda(self) -> float:
        return self._excitation_lambda

    @staticmethod
    def _get_frame_rate_from_sync_file(
            sync_file: SyncFile) -> float:
        timestamps = StimulusTimestamps.from_sync_file(sync_file=sync_file)
        ophys_frame_rate = timestamps.calc_frame_rate()
        return ophys_frame_rate

    @staticmethod
    def _get_targeted_structure_from_lims(ophys_experiment_id: int,
                                          lims_db: PostgresQueryMixin) -> str:
        query = """
                SELECT st.acronym
                FROM ophys_experiments oe
                LEFT JOIN structures st ON st.id = oe.targeted_structure_id
                WHERE oe.id = {};
                """.format(ophys_experiment_id)
        targeted_structure = lims_db.fetchone(query, strict=True)
        return targeted_structure

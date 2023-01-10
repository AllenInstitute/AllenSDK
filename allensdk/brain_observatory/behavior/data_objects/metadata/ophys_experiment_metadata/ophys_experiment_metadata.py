from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects.metadata.ophys_experiment_metadata.field_of_view_shape import FieldOfViewShape  # NOQA
from allensdk.brain_observatory.behavior.data_objects.metadata.ophys_experiment_metadata.imaging_depth import ImagingDepth  # NOQA
from allensdk.brain_observatory.behavior.data_objects.metadata.ophys_experiment_metadata.ophys_container_id import OphysContainerId # NOQA
from allensdk.brain_observatory.behavior.data_objects.metadata.ophys_experiment_metadata.ophys_session_id import OphysSessionId  # NOQA
from allensdk.brain_observatory.behavior.data_objects.metadata.ophys_experiment_metadata.ophys_project_code import OphysProjectCode  # NOQA
from allensdk.brain_observatory.behavior.data_objects.metadata.ophys_experiment_metadata.targeted_imaging_depth import TargetedImagingDepth  # NOQA
from allensdk.core import DataObject, JsonReadableInterface, LimsReadableInterface, NwbReadableInterface  # NOQA
from allensdk.internal.api import PostgresQueryMixin


class OphysExperimentMetadata(DataObject, LimsReadableInterface,
                              JsonReadableInterface, NwbReadableInterface):
    """Container class for ophys experiment metadata"""
    def __init__(self,
                 ophys_experiment_id: int,
                 ophys_session_id: OphysSessionId,
                 ophys_container_id: OphysContainerId,
                 field_of_view_shape: FieldOfViewShape,
                 imaging_depth: ImagingDepth,
                 targeted_imaging_depth: TargetedImagingDepth,
                 project_code: OphysProjectCode = OphysProjectCode()
                 ):
        super().__init__(name='ophys_experiment_metadata', value=None,
                         is_value_self=True)
        self._ophys_experiment_id = ophys_experiment_id
        self._ophys_session_id = ophys_session_id
        self._ophys_container_id = ophys_container_id
        self._field_of_view_shape = field_of_view_shape
        self._imaging_depth = imaging_depth
        self._targeted_imaging_depth = targeted_imaging_depth
        self._project_code = project_code

    @classmethod
    def from_lims(
            cls, ophys_experiment_id: int,
            lims_db: PostgresQueryMixin) -> "OphysExperimentMetadata":
        ophys_session_id = OphysSessionId.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        ophys_container_id = OphysContainerId.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        field_of_view_shape = FieldOfViewShape.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        imaging_depth = ImagingDepth.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        targeted_imaging_depth = TargetedImagingDepth.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        project_code = OphysProjectCode.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)

        return cls(
            ophys_experiment_id=ophys_experiment_id,
            ophys_session_id=ophys_session_id,
            ophys_container_id=ophys_container_id,
            field_of_view_shape=field_of_view_shape,
            imaging_depth=imaging_depth,
            targeted_imaging_depth=targeted_imaging_depth,
            project_code=project_code
        )

    @classmethod
    def from_json(cls, dict_repr: dict) -> "OphysExperimentMetadata":
        ophys_session_id = OphysSessionId.from_json(dict_repr=dict_repr)
        ophys_container_id = OphysContainerId.from_json(
            dict_repr=dict_repr)
        ophys_experiment_id = dict_repr['ophys_experiment_id']
        field_of_view_shape = FieldOfViewShape.from_json(dict_repr=dict_repr)
        imaging_depth = ImagingDepth.from_json(dict_repr=dict_repr)
        targeted_imaging_depth = TargetedImagingDepth.from_json(
            dict_repr=dict_repr
        )

        return OphysExperimentMetadata(
            ophys_experiment_id=ophys_experiment_id,
            ophys_session_id=ophys_session_id,
            ophys_container_id=ophys_container_id,
            field_of_view_shape=field_of_view_shape,
            imaging_depth=imaging_depth,
            targeted_imaging_depth=targeted_imaging_depth
        )

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "OphysExperimentMetadata":
        ophys_experiment_id = int(nwbfile.identifier)
        ophys_session_id = OphysSessionId.from_nwb(nwbfile=nwbfile)
        ophys_container_id = OphysContainerId.from_nwb(
            nwbfile=nwbfile)
        field_of_view_shape = FieldOfViewShape.from_nwb(nwbfile=nwbfile)
        imaging_depth = ImagingDepth.from_nwb(nwbfile=nwbfile)
        targeted_imaging_depth = TargetedImagingDepth.from_nwb(
            nwbfile=nwbfile)
        project_code = OphysProjectCode.from_nwb(nwbfile=nwbfile)

        return OphysExperimentMetadata(
            ophys_experiment_id=ophys_experiment_id,
            ophys_session_id=ophys_session_id,
            ophys_container_id=ophys_container_id,
            field_of_view_shape=field_of_view_shape,
            imaging_depth=imaging_depth,
            targeted_imaging_depth=targeted_imaging_depth,
            project_code=project_code
        )

    @property
    def ophys_container_id(self) -> int:
        # TODO: Remove upon updated VBO release.
        if self._ophys_container_id is None:
            return None
        return self._ophys_container_id.value

    @property
    def field_of_view_shape(self) -> FieldOfViewShape:
        return self._field_of_view_shape

    @property
    def imaging_depth(self) -> int:
        # TODO: Remove upon updated VBO release.
        if self._ophys_container_id is None:
            return None
        return self._imaging_depth.value

    @property
    def targeted_imaging_depth(self) -> int:
        # TODO: Remove upon updated VBO release.
        if self._targeted_imaging_depth is None:
            return None
        return self._targeted_imaging_depth.value

    @property
    def ophys_experiment_id(self) -> int:
        return self._ophys_experiment_id

    @property
    def ophys_session_id(self) -> int:
        # TODO this is at the wrong layer of abstraction.
        #  Should be at ophys session level
        #  (need to create ophys session class)
        return self._ophys_session_id.value

    @property
    def project_code(self) -> str:
        return self._project_code.value

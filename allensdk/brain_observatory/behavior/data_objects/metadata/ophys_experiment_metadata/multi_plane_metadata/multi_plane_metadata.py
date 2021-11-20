from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.experiment_container_id import \
    ExperimentContainerId
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.field_of_view_shape import \
    FieldOfViewShape
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.imaging_depth import \
    ImagingDepth
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.multi_plane_metadata \
    .imaging_plane_group import \
    ImagingPlaneGroup
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.ophys_experiment_metadata import \
    OphysExperimentMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.ophys_session_id import \
    OphysSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.project_code import \
    ProjectCode
from allensdk.internal.api import PostgresQueryMixin


class MultiplaneMetadata(OphysExperimentMetadata):
    def __init__(self,
                 ophys_experiment_id: int,
                 ophys_session_id: OphysSessionId,
                 experiment_container_id: ExperimentContainerId,
                 field_of_view_shape: FieldOfViewShape,
                 imaging_depth: ImagingDepth,
                 imaging_plane_group: ImagingPlaneGroup,
                 project_code: ProjectCode):
        super().__init__(
            ophys_experiment_id=ophys_experiment_id,
            ophys_session_id=ophys_session_id,
            experiment_container_id=experiment_container_id,
            field_of_view_shape=field_of_view_shape,
            imaging_depth=imaging_depth,
            project_code=project_code
        )
        self._imaging_plane_group = imaging_plane_group

    @classmethod
    def from_lims(
            cls, ophys_experiment_id: int,
            lims_db: PostgresQueryMixin) -> "MultiplaneMetadata":
        ophys_experiment_metadata = OphysExperimentMetadata.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        imaging_plane_group = ImagingPlaneGroup.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        return cls(
            ophys_experiment_id=ophys_experiment_metadata.ophys_experiment_id,
            ophys_session_id=ophys_experiment_metadata._ophys_session_id,
            experiment_container_id=ophys_experiment_metadata._experiment_container_id,     # noqa E501
            field_of_view_shape=ophys_experiment_metadata._field_of_view_shape,
            imaging_depth=ophys_experiment_metadata._imaging_depth,
            project_code=ophys_experiment_metadata._project_code,
            imaging_plane_group=imaging_plane_group
        )

    @classmethod
    def from_json(cls, dict_repr: dict) -> "MultiplaneMetadata":
        ophys_experiment_metadata = super().from_json(dict_repr=dict_repr)
        imaging_plane_group = ImagingPlaneGroup.from_json(dict_repr=dict_repr)
        return cls(
            ophys_experiment_id=ophys_experiment_metadata.ophys_experiment_id,
            ophys_session_id=ophys_experiment_metadata._ophys_session_id,
            experiment_container_id=ophys_experiment_metadata._experiment_container_id, # noqa E501
            field_of_view_shape=ophys_experiment_metadata._field_of_view_shape,
            imaging_depth=ophys_experiment_metadata._imaging_depth,
            project_code=ophys_experiment_metadata._project_code,
            imaging_plane_group=imaging_plane_group
        )

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "MultiplaneMetadata":
        ophys_experiment_metadata = super().from_nwb(nwbfile=nwbfile)
        imaging_plane_group = ImagingPlaneGroup.from_nwb(nwbfile=nwbfile)
        return cls(
            ophys_experiment_id=ophys_experiment_metadata.ophys_experiment_id,
            ophys_session_id=ophys_experiment_metadata._ophys_session_id,
            experiment_container_id=ophys_experiment_metadata._experiment_container_id, # noqa E501
            field_of_view_shape=ophys_experiment_metadata._field_of_view_shape,
            imaging_depth=ophys_experiment_metadata._imaging_depth,
            project_code=ophys_experiment_metadata._project_code,
            imaging_plane_group=imaging_plane_group
        )

    @property
    def imaging_plane_group(self) -> int:
        return self._imaging_plane_group.plane_group

    @property
    def imaging_plane_group_count(self) -> int:
        # TODO this is at the wrong level of abstraction.
        #  It is an attribute of the session, not the experiment.
        #  Currently, an Ophys Session metadata abstraction doesn't exist
        return self._imaging_plane_group.plane_group_count

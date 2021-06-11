from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects.base\
    .readable_interfaces.internal_readable_interface import \
    InternalReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .readable_interfaces.json_readable_interface import \
    JsonReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .readable_interfaces.nwb_readable_interface import \
    NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.ophys_experiment_metadata import \
    OphysExperimentMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.emission_lambda import \
    EmissionLambda
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.excitation_lambda import \
    ExcitationLambda
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.experiment_container_id import \
    ExperimentContainerId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.field_of_view_shape import \
    FieldOfViewShape
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.imaging_depth import \
    ImagingDepth
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.mesoscope_experiment_metadata\
    .imaging_plane_group import \
    ImagingPlaneGroup
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.ophys_frame_rate import \
    OphysFrameRate
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.project_code import \
    ProjectCode
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.targeted_structure import \
    TargetedStructure
from allensdk.internal.api import PostgresQueryMixin


class MesoscopeExperimentMetadata(OphysExperimentMetadata):
    def __init__(self,
                 experiment_container_id: ExperimentContainerId,
                 emission_lambda: EmissionLambda,
                 excitation_lambda: ExcitationLambda,
                 field_of_view_shape: FieldOfViewShape,
                 imaging_depth: ImagingDepth,
                 imaging_plane_group: ImagingPlaneGroup,
                 ophys_frame_rate: OphysFrameRate,
                 project_code: ProjectCode,
                 targeted_structure: TargetedStructure):
        super().__init__(
            experiment_container_id=experiment_container_id,
            emission_lambda=emission_lambda,
            excitation_lambda=excitation_lambda,
            field_of_view_shape=field_of_view_shape,
            imaging_depth=imaging_depth,
            ophys_frame_rate=ophys_frame_rate,
            project_code=project_code,
            targeted_structure=targeted_structure
        )
        self._imaging_plane_group = imaging_plane_group

    @classmethod
    def from_internal(
            cls, ophys_experiment_id: int,
            lims_db: PostgresQueryMixin) -> "MesoscopeExperimentMetadata":
        ophys_experiment_metadata = OphysExperimentMetadata.from_internal(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        imaging_plane_group = ImagingPlaneGroup.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        return cls(
            experiment_container_id=
            ophys_experiment_metadata._experiment_container_id,
            emission_lambda=ophys_experiment_metadata._emission_lambda,
            excitation_lambda=ophys_experiment_metadata._excitation_lambda,
            field_of_view_shape=ophys_experiment_metadata._field_of_view_shape,
            imaging_depth=ophys_experiment_metadata._imaging_depth,
            ophys_frame_rate=ophys_experiment_metadata._ophys_frame_rate,
            project_code=ophys_experiment_metadata._project_code,
            targeted_structure=ophys_experiment_metadata._targeted_structure,
            imaging_plane_group=imaging_plane_group
        )

    @classmethod
    def from_json(cls, dict_repr: dict) -> "MesoscopeExperimentMetadata":
        ophys_experiment_metadata = super().from_json(dict_repr=dict_repr)
        imaging_plane_group = ImagingPlaneGroup.from_json(dict_repr=dict_repr)
        return cls(
            experiment_container_id=
            ophys_experiment_metadata._experiment_container_id,
            emission_lambda=ophys_experiment_metadata._emission_lambda,
            excitation_lambda=ophys_experiment_metadata._excitation_lambda,
            field_of_view_shape=ophys_experiment_metadata._field_of_view_shape,
            imaging_depth=ophys_experiment_metadata._imaging_depth,
            imaging_plane_group=imaging_plane_group
        )

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "MesoscopeExperimentMetadata":
        ophys_experiment_metadata = super().from_nwb(nwbfile=nwbfile)
        imaging_plane_group = ImagingPlaneGroup.from_nwb(nwbfile=nwbfile)
        return cls(
            experiment_container_id=
            ophys_experiment_metadata._experiment_container_id,
            emission_lambda=ophys_experiment_metadata._emission_lambda,
            excitation_lambda=ophys_experiment_metadata._excitation_lambda,
            field_of_view_shape=ophys_experiment_metadata._field_of_view_shape,
            imaging_depth=ophys_experiment_metadata._imaging_depth,
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

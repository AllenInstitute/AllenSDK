from typing import Optional

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import SyncFile
from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
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
    .ophys_experiment_metadata.emission_lambda import \
    EmissionLambda
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
    .ophys_experiment_metadata.imaging_plane import \
    ImagingPlane
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.project_code import \
    ProjectCode
from allensdk.internal.api import PostgresQueryMixin


class OphysExperimentMetadata(DataObject, InternalReadableInterface,
                              JsonReadableInterface, NwbReadableInterface):
    """Container class for behavior ophys metadata"""
    # def __init__(self, extractor: BehaviorOphysDataExtractorBase,
    #              stimulus_timestamps: np.ndarray,
    #              ophys_timestamps: np.ndarray,
    #              behavior_stimulus_file: dict):
    def __init__(self,
                 experiment_container_id: ExperimentContainerId,
                 imaging_plane: ImagingPlane,
                 emission_lambda: EmissionLambda,
                 field_of_view_shape: FieldOfViewShape,
                 imaging_depth: ImagingDepth,
                 project_code: ProjectCode):
        super().__init__(name='ophys_experiment_metadata', value=self)
        self._experiment_container_id = experiment_container_id
        self._imaging_plane = imaging_plane
        self._emission_lambda = emission_lambda
        self._field_of_view_shape = field_of_view_shape
        self._imaging_depth = imaging_depth
        self._project_code = project_code

        # project_code needs to be excluded from comparison
        # since it's only exposed internally
        self._exclude_from_equals = {'project_code'}

    @classmethod
    def from_internal(
            cls, ophys_experiment_id: int,
            lims_db: PostgresQueryMixin) -> "OphysExperimentMetadata":
        experiment_container_id = ExperimentContainerId.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        imaging_plane = ImagingPlane.from_internal(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        emission_lambda = EmissionLambda(emission_lambda=520.0)
        field_of_view_shape = FieldOfViewShape.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        imaging_depth = ImagingDepth.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        project_code = ProjectCode.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)

        return cls(
            emission_lambda=emission_lambda,
            imaging_plane=imaging_plane,
            experiment_container_id=experiment_container_id,
            field_of_view_shape=field_of_view_shape,
            imaging_depth=imaging_depth,
            project_code=project_code,
        )

    @classmethod
    def from_json(cls, dict_repr: dict) -> "OphysExperimentMetadata":
        pass

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "OphysExperimentMetadata":
        pass

    @property
    def emission_lambda(self) -> float:
        return self._emission_lambda.value

    @property
    def imaging_plane(self) -> ImagingPlane:
        return self._imaging_plane

    # TODO rename to ophys_container_id
    @property
    def experiment_container_id(self) -> int:
        return self._experiment_container_id.value

    @property
    def field_of_view_shape(self) -> FieldOfViewShape:
        return self._field_of_view_shape

    @property
    def imaging_depth(self) -> int:
        return self._imaging_depth.value

    @property
    def ophys_experiment_id(self) -> int:
        # TODO remove this. Should be at Ophys experiment class level
        return 0

    @property
    def ophys_session_id(self) -> int:
        # TODO this is at the wrong layer of abstraction.
        #  Should be at ophys session level
        return 0

    @property
    def project_code(self) -> Optional[str]:
        return self._project_code.value

    def to_dict(self) -> dict:
        """Returns dict representation of all properties in class"""
        vars_ = vars(OphysExperimentMetadata)
        d = self._get_properties(vars_=vars_)
        return {**super().to_dict(), **d}

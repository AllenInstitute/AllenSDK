from typing import Optional

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, NwbReadableInterface, \
    LimsReadableInterface
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
    .ophys_experiment_metadata.ophys_session_id import \
    OphysSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.project_code import \
    ProjectCode
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.nwb import load_pynwb_extension
from allensdk.brain_observatory.behavior.schemas import OphysMetadataSchema


class OphysExperimentMetadata(DataObject, LimsReadableInterface,
                              JsonReadableInterface, NwbReadableInterface):
    """Container class for ophys experiment metadata"""
    def __init__(self,
                 ophys_experiment_id: int,
                 ophys_session_id: OphysSessionId,
                 field_of_view_shape: FieldOfViewShape,
                 imaging_depth: ImagingDepth,
                 project_code: Optional[ProjectCode] = None):
        super().__init__(name='ophys_experiment_metadata', value=self)
        self._ophys_experiment_id = ophys_experiment_id
        self._ophys_session_id = ophys_session_id
        self._field_of_view_shape = field_of_view_shape
        self._imaging_depth = imaging_depth
        self._project_code = project_code

        # project_code needs to be excluded from comparison
        # since it's only exposed internally
        self._exclude_from_equals = {'project_code'}

    @classmethod
    def from_lims(
            cls, ophys_experiment_id: int,
            lims_db: PostgresQueryMixin) -> "OphysExperimentMetadata":
        ophys_session_id = OphysSessionId.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        field_of_view_shape = FieldOfViewShape.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        imaging_depth = ImagingDepth.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        project_code = ProjectCode.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)

        return cls(
            ophys_experiment_id=ophys_experiment_id,
            ophys_session_id=ophys_session_id,
            field_of_view_shape=field_of_view_shape,
            imaging_depth=imaging_depth,
            project_code=project_code
        )

    @classmethod
    def from_json(cls, dict_repr: dict) -> "OphysExperimentMetadata":
        ophys_session_id = OphysSessionId.from_json(dict_repr=dict_repr)
        ophys_experiment_id = dict_repr['ophys_experiment_id']
        field_of_view_shape = FieldOfViewShape.from_json(dict_repr=dict_repr)
        imaging_depth = ImagingDepth.from_json(dict_repr=dict_repr)

        return OphysExperimentMetadata(
            ophys_experiment_id=ophys_experiment_id,
            ophys_session_id=ophys_session_id,
            field_of_view_shape=field_of_view_shape,
            imaging_depth=imaging_depth
        )

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "OphysExperimentMetadata":
        ophys_experiment_id = int(nwbfile.identifier)
        ophys_session_id = OphysSessionId.from_nwb(nwbfile=nwbfile)
        field_of_view_shape = FieldOfViewShape.from_nwb(nwbfile=nwbfile)
        imaging_depth = ImagingDepth.from_nwb(nwbfile=nwbfile)

        return OphysExperimentMetadata(
            ophys_experiment_id=ophys_experiment_id,
            ophys_session_id=ophys_session_id,
            field_of_view_shape=field_of_view_shape,
            imaging_depth=imaging_depth
        )

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        extension = load_pynwb_extension(OphysMetadataSchema,
                                         'ndx-aibs-behavior-ophys')
        nwb_metadata = extension(
            name='metadata',
            ophys_experiment_id =self._ophys_experiment_id,
            ophys_session_id = self._ophys_session_id.value,
            experiment_container_id = 0,
            field_of_view_height = self._field_of_view_shape.value._height,
            field_of_view_width = self._field_of_view_shape.value._width,
            imaging_depth = self._imaging_depth.value,
            imaging_plane_group = -1,
            imaging_plane_group_count = 0
        )
        device_config = {
                "name": "MESO.2",
                "description": "Allen Brain Observatory - Mesoscope 2P Rig"
        }
        nwbfile.create_device(**device_config)
        nwbfile.add_lab_meta_data(nwb_metadata)

        return nwbfile

    @property
    def field_of_view_shape(self) -> FieldOfViewShape:
        return self._field_of_view_shape

    @property
    def imaging_depth(self) -> int:
        return self._imaging_depth.value

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
    def project_code(self) -> Optional[str]:
        if self._project_code is None:
            pc = self._project_code
        else:
            pc = self._project_code.value
        return pc

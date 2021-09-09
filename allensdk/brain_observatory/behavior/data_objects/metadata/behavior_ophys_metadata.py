from typing import Union

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, NwbReadableInterface, \
    LimsReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.multi_plane_metadata\
    .multi_plane_metadata import \
    MultiplaneMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.ophys_experiment_metadata import \
    OphysExperimentMetadata
from allensdk.brain_observatory.behavior.schemas import \
    OphysBehaviorMetadataSchema
from allensdk.brain_observatory.nwb import load_pynwb_extension
from allensdk.internal.api import PostgresQueryMixin


class BehaviorOphysMetadata(DataObject, LimsReadableInterface,
                            JsonReadableInterface, NwbReadableInterface,
                            NwbWritableInterface):
    def __init__(self, behavior_metadata: BehaviorMetadata,
                 ophys_metadata: Union[OphysExperimentMetadata,
                                       MultiplaneMetadata]):
        super().__init__(name='behavior_ophys_metadata', value=self)

        self._behavior_metadata = behavior_metadata
        self._ophys_metadata = ophys_metadata

    @property
    def behavior_metadata(self) -> BehaviorMetadata:
        return self._behavior_metadata

    @property
    def ophys_metadata(self) -> Union["OphysExperimentMetadata",
                                      "MultiplaneMetadata"]:
        return self._ophys_metadata

    @classmethod
    def from_lims(cls, ophys_experiment_id: int,
                  lims_db: PostgresQueryMixin,
                  is_multiplane=False) -> "BehaviorOphysMetadata":
        """

        Parameters
        ----------
        ophys_experiment_id
        lims_db
        is_multiplane
            Whether to fetch metadata for an experiment that is part of a
            container containing multiple imaging planes
        """
        behavior_session_id = BehaviorSessionId.from_lims(
            ophys_experiment_id=ophys_experiment_id, db=lims_db)

        behavior_metadata = BehaviorMetadata.from_lims(
            behavior_session_id=behavior_session_id, lims_db=lims_db)

        if is_multiplane:
            ophys_metadata = MultiplaneMetadata.from_lims(
                ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        else:
            ophys_metadata = OphysExperimentMetadata.from_lims(
                ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)

        return cls(behavior_metadata=behavior_metadata,
                   ophys_metadata=ophys_metadata)

    @classmethod
    def from_json(cls, dict_repr: dict,
                  is_multiplane=False) -> "BehaviorOphysMetadata":
        """

        Parameters
        ----------
        dict_repr
        is_multiplane
            Whether to fetch metadata for an experiment that is part of a
            container containing multiple imaging planes

        Returns
        -------

        """
        behavior_metadata = BehaviorMetadata.from_json(dict_repr=dict_repr)

        if is_multiplane:
            ophys_metadata = MultiplaneMetadata.from_json(
                dict_repr=dict_repr)
        else:
            ophys_metadata = OphysExperimentMetadata.from_json(
                dict_repr=dict_repr)

        return cls(behavior_metadata=behavior_metadata,
                   ophys_metadata=ophys_metadata)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile,
                 is_multiplane=False) -> "BehaviorOphysMetadata":
        """

        Parameters
        ----------
        nwbfile
        is_multiplane
            Whether to fetch metadata for an experiment that is part of a
            container containing multiple imaging planes
        """
        behavior_metadata = BehaviorMetadata.from_nwb(nwbfile=nwbfile)

        if is_multiplane:
            ophys_metadata = MultiplaneMetadata.from_nwb(
                nwbfile=nwbfile)
        else:
            ophys_metadata = OphysExperimentMetadata.from_nwb(
                nwbfile=nwbfile)

        return cls(behavior_metadata=behavior_metadata,
                   ophys_metadata=ophys_metadata)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        self._behavior_metadata.subject_metadata.to_nwb(nwbfile=nwbfile)
        self._behavior_metadata.equipment.to_nwb(nwbfile=nwbfile)

        nwb_extension = load_pynwb_extension(
            OphysBehaviorMetadataSchema, 'ndx-aibs-behavior-ophys')

        behavior_meta = self._behavior_metadata
        ophys_meta = self._ophys_metadata

        if isinstance(ophys_meta, MultiplaneMetadata):
            imaging_plane_group = ophys_meta.imaging_plane_group
            imaging_plane_group_count = ophys_meta.imaging_plane_group_count
        else:
            imaging_plane_group_count = 0
            imaging_plane_group = -1

        nwb_metadata = nwb_extension(
            name='metadata',
            ophys_session_id=ophys_meta.ophys_session_id,
            field_of_view_width=ophys_meta.field_of_view_shape.width,
            field_of_view_height=ophys_meta.field_of_view_shape.height,
            imaging_plane_group=imaging_plane_group,
            imaging_plane_group_count=imaging_plane_group_count,
            stimulus_frame_rate=behavior_meta.stimulus_frame_rate,
            experiment_container_id=ophys_meta.experiment_container_id,
            ophys_experiment_id=ophys_meta.ophys_experiment_id,
            session_type=behavior_meta.session_type,
            equipment_name=behavior_meta.equipment.value,
            imaging_depth=ophys_meta.imaging_depth,
            behavior_session_uuid=str(behavior_meta.behavior_session_uuid),
            behavior_session_id=behavior_meta.behavior_session_id
        )
        nwbfile.add_lab_meta_data(nwb_metadata)

        return nwbfile

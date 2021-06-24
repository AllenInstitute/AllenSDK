import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_objects.cell_specimen_table \
    import \
    CellSpecimenTable
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.session_type import \
    SessionType
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_ophys_metadata import \
    BehaviorOphysMetadata
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
    .ophys_experiment_metadata.mesoscope_experiment_metadata\
    .imaging_plane_group import \
    ImagingPlaneGroup
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.mesoscope_experiment_metadata\
    .mesoscope_experiment_metadata import \
    MesoscopeExperimentMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.ophys_experiment_metadata import \
    OphysExperimentMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.ophys_session_id import \
    OphysSessionId
from allensdk.test.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.test_behavior_metadata import \
    TestBehaviorMetadata


class TestBehaviorOphysMetadata:
    def setup_method(self, method):
        self.meta = self._get_meta()
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier=str(self.meta.ophys_metadata.ophys_experiment_id),
            session_start_time=self.meta.behavior_metadata.date_of_acquisition
        )

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_write_nwb_read_no_cell_specimen_table(
            self, roundtrip, data_object_roundtrip_fixture):
        self.meta.to_nwb(nwbfile=self.nwbfile)

        with pytest.raises(RuntimeError):
            if roundtrip:
                data_object_roundtrip_fixture(
                    nwbfile=self.nwbfile,
                    data_object_cls=BehaviorOphysMetadata)
            else:
                self.meta.from_nwb(nwbfile=self.nwbfile)

    @pytest.mark.parametrize('meso', [True, False])
    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip, cell_specimen_table,
                            data_object_roundtrip_fixture, meso):
        if meso:
            self.meta = self._get_mesoscope_meta()

        self.meta.to_nwb(nwbfile=self.nwbfile)

        cell_specimen_table = pd.DataFrame(cell_specimen_table)
        cell_specimen_table = CellSpecimenTable(
            cell_specimen_table=cell_specimen_table)
        cell_specimen_table.to_nwb(nwbfile=self.nwbfile, meta=self.meta)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=BehaviorOphysMetadata)
        else:
            obt = self.meta.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.meta

    @staticmethod
    def _get_meta():
        ophys_meta = OphysExperimentMetadata(
            ophys_experiment_id=1234,
            ophys_session_id=OphysSessionId(session_id=999),
            experiment_container_id=ExperimentContainerId(
                experiment_container_id=5678),
            imaging_plane=ImagingPlane(
                ophys_frame_rate=31.0,
                targeted_structure='VISp',
                excitation_lambda=1.0
            ),
            emission_lambda=EmissionLambda(emission_lambda=1.0),
            field_of_view_shape=FieldOfViewShape(width=4, height=4),
            imaging_depth=ImagingDepth(imaging_depth=375)
        )

        behavior_metadata = TestBehaviorMetadata()
        behavior_metadata.setup_class()
        behavior_metadata = behavior_metadata.meta
        return BehaviorOphysMetadata(
            behavior_metadata=behavior_metadata,
            ophys_metadata=ophys_meta
        )

    def _get_mesoscope_meta(self):
        bo_meta = self.meta
        bo_meta.behavior_metadata._session_type = \
            SessionType(session_type='MESO.1')
        ophys_experiment_metadata = bo_meta.ophys_metadata

        imaging_plane_group = ImagingPlaneGroup(plane_group_count=5,
                                                plane_group=0)
        meso_meta = MesoscopeExperimentMetadata(
            ophys_experiment_id=ophys_experiment_metadata.ophys_experiment_id,
            ophys_session_id=ophys_experiment_metadata._ophys_session_id,
            experiment_container_id=
            ophys_experiment_metadata._experiment_container_id,
            emission_lambda=ophys_experiment_metadata._emission_lambda,
            imaging_plane=ophys_experiment_metadata._imaging_plane,
            field_of_view_shape=ophys_experiment_metadata._field_of_view_shape,
            imaging_depth=ophys_experiment_metadata._imaging_depth,
            project_code=ophys_experiment_metadata._project_code,
            imaging_plane_group=imaging_plane_group
        )
        return BehaviorOphysMetadata(
            behavior_metadata=bo_meta.behavior_metadata,
            ophys_metadata=meso_meta
        )

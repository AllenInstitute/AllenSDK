import datetime
import json
from pathlib import Path
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.equipment import \
    Equipment
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_ophys_metadata import \
    BehaviorOphysMetadata
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
    .ophys_experiment_metadata.multi_plane_metadata\
    .imaging_plane_group import \
    ImagingPlaneGroup
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.multi_plane_metadata\
    .multi_plane_metadata import \
    MultiplaneMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.ophys_experiment_metadata import \
    OphysExperimentMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.ophys_session_id import \
    OphysSessionId
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.internal.api import db_connection_creator
from allensdk.test.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.test_behavior_metadata import \
    TestBehaviorMetadata


class TestBOM:
    @classmethod
    def setup_class(cls):
        cls.meta = cls._get_meta()

    def setup_method(self, method):
        self.meta = self._get_meta()

    @staticmethod
    def _get_meta():
        ophys_meta = OphysExperimentMetadata(
            ophys_experiment_id=1234,
            ophys_session_id=OphysSessionId(session_id=999),
            experiment_container_id=ExperimentContainerId(
                experiment_container_id=5678),
            field_of_view_shape=FieldOfViewShape(width=4, height=4),
            imaging_depth=ImagingDepth(imaging_depth=375)
        )

        behavior_metadata = TestBehaviorMetadata()
        behavior_metadata.setup_class()
        return BehaviorOphysMetadata(
            behavior_metadata=behavior_metadata.meta,
            ophys_metadata=ophys_meta
        )

    def _get_multiplane_meta(self):
        bo_meta = self.meta
        bo_meta.behavior_metadata._equipment = \
            Equipment(equipment_name='MESO.1')
        ophys_experiment_metadata = bo_meta.ophys_metadata

        imaging_plane_group = ImagingPlaneGroup(plane_group_count=5,
                                                plane_group=0)
        multiplane_meta = MultiplaneMetadata(
            ophys_experiment_id=ophys_experiment_metadata.ophys_experiment_id,
            ophys_session_id=ophys_experiment_metadata._ophys_session_id,
            experiment_container_id=ophys_experiment_metadata._experiment_container_id, # noqa E501
            field_of_view_shape=ophys_experiment_metadata._field_of_view_shape,
            imaging_depth=ophys_experiment_metadata._imaging_depth,
            project_code=ophys_experiment_metadata._project_code,
            imaging_plane_group=imaging_plane_group
        )
        return BehaviorOphysMetadata(
            behavior_metadata=bo_meta.behavior_metadata,
            ophys_metadata=multiplane_meta
        )


class TestInternal(TestBOM):
    @classmethod
    def setup_method(self, method):
        marks = getattr(method, 'pytestmark', None)
        if marks:
            marks = [m.name for m in marks]

            # Will only create a dbconn if the test requires_bamboo
            if 'requires_bamboo' in marks:
                self.dbconn = db_connection_creator(
                    fallback_credentials=LIMS_DB_CREDENTIAL_MAP)

    @pytest.mark.requires_bamboo
    @pytest.mark.parametrize('meso', [True, False])
    def test_from_lims(self, meso):
        if meso:
            ophys_experiment_id = 951980471
        else:
            ophys_experiment_id = 994278291
        bom = BehaviorOphysMetadata.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=self.dbconn,
            is_multiplane=meso)

        if meso:
            assert isinstance(bom.ophys_metadata,
                              MultiplaneMetadata)
            assert bom.ophys_metadata.imaging_depth == 150
            assert bom.behavior_metadata.session_type == 'OPHYS_1_images_A'
            assert bom.behavior_metadata.subject_metadata.reporter_line == \
                   'Ai148(TIT2L-GC6f-ICL-tTA2)'
            assert bom.behavior_metadata.subject_metadata.driver_line == \
                   ['Sst-IRES-Cre']
            assert bom.behavior_metadata.subject_metadata.mouse_id == 457841
            assert bom.behavior_metadata.subject_metadata.full_genotype == \
                   'Sst-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt'
            assert bom.behavior_metadata.subject_metadata.age_in_days == 233
            assert bom.behavior_metadata.subject_metadata.sex == 'F'
        else:
            assert isinstance(bom.ophys_metadata, OphysExperimentMetadata)
            assert bom.ophys_metadata.imaging_depth == 175
            assert bom.behavior_metadata.session_type == 'OPHYS_4_images_A'
            assert bom.behavior_metadata.subject_metadata.reporter_line == \
                   'Ai93(TITL-GCaMP6f)'
            assert bom.behavior_metadata.subject_metadata.driver_line == \
                   ['Camk2a-tTA', 'Slc17a7-IRES2-Cre']
            assert bom.behavior_metadata.subject_metadata.mouse_id == 491060
            assert bom.behavior_metadata.subject_metadata.full_genotype == \
                   'Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt'
            assert bom.behavior_metadata.subject_metadata.age_in_days == 130
            assert bom.behavior_metadata.subject_metadata.sex == 'M'


class TestJson(TestBOM):
    @classmethod
    def setup_method(self, method):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir.parent / 'test_data'
        with open(test_data_dir / 'test_input.json') as f:
            dict_repr = json.load(f)
        dict_repr = dict_repr['session_data']
        dict_repr['sync_file'] = str(test_data_dir / 'sync.h5')
        dict_repr['behavior_stimulus_file'] = str(test_data_dir /
                                                  'behavior_stimulus_file.pkl')
        dict_repr['dff_file'] = str(test_data_dir / 'demix_file.h5')
        self.dict_repr = dict_repr

    @pytest.mark.parametrize('meso', [True, False])
    def test_from_json(self, meso):
        if meso:
            self.dict_repr['rig_name'] = 'MESO.1'
        bom = BehaviorOphysMetadata.from_json(dict_repr=self.dict_repr,
                                              is_multiplane=meso)

        if meso:
            assert isinstance(bom.ophys_metadata, MultiplaneMetadata)
        else:
            assert isinstance(bom.ophys_metadata, OphysExperimentMetadata)


class TestNWB(TestBOM):
    def setup_method(self, method):
        self.meta = self._get_meta()
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier=str(self.meta.ophys_metadata.ophys_experiment_id),
            session_start_time=datetime.datetime.now()
        )

    @pytest.mark.parametrize('meso', [True, False])
    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture, meso):
        if meso:
            self.meta = self._get_multiplane_meta()

        self.meta.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=BehaviorOphysMetadata,
                is_multiplane=meso)
        else:
            obt = self.meta.from_nwb(nwbfile=self.nwbfile,
                                     is_multiplane=meso)

        assert obt == self.meta

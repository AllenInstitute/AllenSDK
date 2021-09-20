import json
from datetime import datetime
from pathlib import Path

import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_objects.projections import \
    Projections
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.internal.api import db_connection_creator


class TestLims:
    @classmethod
    def setup_class(cls):
        cls.ophys_experiment_id = 994278291

        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        cls.expected_max = Projections._from_filepath(
            filepath=str(test_data_dir / 'max_projection.png'),
            pixel_size=.78125)

        cls.expected_avg = Projections._from_filepath(
            filepath=str(test_data_dir / 'avg_projection.png'),
            pixel_size=.78125)

    def setup_method(self, method):
        marks = getattr(method, 'pytestmark', None)
        if marks:
            marks = [m.name for m in marks]

            # Will only create a dbconn if the test requires_bamboo
            if 'requires_bamboo' in marks:
                self.dbconn = db_connection_creator(
                    fallback_credentials=LIMS_DB_CREDENTIAL_MAP)

    @pytest.mark.requires_bamboo
    def test_from_lims(self):
        projections = Projections.from_lims(
            ophys_experiment_id=self.ophys_experiment_id, lims_db=self.dbconn)

        assert projections.max_projection == self.expected_max
        assert projections.avg_projection == self.expected_avg


class TestJson:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'
        with open(test_data_dir / 'test_input.json') as f:
            dict_repr = json.load(f)
        dict_repr = dict_repr['session_data']
        dict_repr['max_projection_file'] = test_data_dir / \
            dict_repr['max_projection_file']
        dict_repr['average_intensity_projection_image_file'] = \
            test_data_dir / \
            dict_repr['average_intensity_projection_image_file']

        cls.expected_max = Projections._from_filepath(
            filepath=str(test_data_dir / 'max_projection.png'),
            pixel_size=.78125)

        cls.expected_avg = Projections._from_filepath(
            filepath=str(test_data_dir / 'avg_projection.png'),
            pixel_size=.78125)

        cls.dict_repr = dict_repr

    def test_from_json(self):
        projections = Projections.from_json(dict_repr=self.dict_repr)

        assert projections.max_projection == self.expected_max
        assert projections.avg_projection == self.expected_avg


class TestNWB:
    @classmethod
    def setup_class(cls):
        tj = TestJson()
        tj.setup_class()
        cls.projections = Projections.from_json(
            dict_repr=tj.dict_repr)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self.projections.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=Projections)
        else:
            obt = self.projections.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.projections

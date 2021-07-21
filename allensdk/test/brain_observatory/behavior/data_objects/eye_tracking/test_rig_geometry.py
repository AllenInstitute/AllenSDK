import json

from datetime import datetime
from pathlib import Path

import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_objects.eye_tracking\
    .rig_geometry import \
    RigGeometry
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestFromLims(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.ophys_experiment_id = 994278291

        dir = Path(__file__).parent.parent.resolve()
        test_data_dir = dir / 'test_data'

        with open(test_data_dir / 'eye_tracking_rig_geometry.json') as f:
            x = json.load(f)
            x = x['rig_geometry']
            cls.expected = RigGeometry(**x)

    @pytest.mark.requires_bamboo
    def test_from_lims(self):
        rg = RigGeometry.from_lims(
            ophys_experiment_id=self.ophys_experiment_id, lims_db=self.dbconn)
        assert rg == self.expected


class TestFromJson(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.ophys_experiment_id = 994278291

        dir = Path(__file__).parent.parent.resolve()
        test_data_dir = dir / 'test_data'

        with open(test_data_dir / 'eye_tracking_rig_geometry.json') as f:
            x = json.load(f)
            x = x['rig_geometry']
            cls.expected = RigGeometry(**x)

    @pytest.mark.requires_bamboo
    def test_from_json(self):
        dict_repr = {'eye_tracking_rig_geometry':
                     self.expected.to_dict()['rig_geometry']}
        rg = RigGeometry.from_json(dict_repr=dict_repr)
        assert rg == self.expected


class TestNWB:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.parent.resolve()
        cls.test_data_dir = dir / 'test_data'

        with open(cls.test_data_dir / 'eye_tracking_rig_geometry.json') as f:
            x = json.load(f)
            x = x['rig_geometry']
            cls.rig_geometry = RigGeometry(**x)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self.rig_geometry.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=RigGeometry)
        else:
            obt = RigGeometry.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.rig_geometry

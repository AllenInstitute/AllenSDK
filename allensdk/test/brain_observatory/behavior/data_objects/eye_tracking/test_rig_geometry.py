import json
import pandas as pd

from datetime import datetime
from pathlib import Path

import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_objects.eye_tracking \
    .rig_geometry import \
    RigGeometry, Coordinates
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
            x = {'eye_tracking_rig_geometry': x}
            cls.expected = RigGeometry.from_json(dict_repr=x)

    @pytest.mark.requires_bamboo
    def test_from_lims(self):
        rg = RigGeometry.from_lims(
            ophys_experiment_id=self.ophys_experiment_id, lims_db=self.dbconn)
        assert rg == self.expected

    @pytest.mark.requires_bamboo
    def test_rig_geometry_newer_than_experiment(self):
        """
        This test ensures that if the experiment date_of_acquisition
        is before a rig activate_date that it is not returned as the rig
        used for the experiment
        """
        # This experiment has rig config more recent than the
        # experiment date_of_acquisition
        ophys_experiment_id = 521405260

        rg = RigGeometry.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=self.dbconn)
        expected = RigGeometry(
            camera_position_mm=Coordinates(x=130.0, y=0.0, z=0.0),
            led_position=Coordinates(x=265.1, y=-39.3, z=1.0),
            monitor_position_mm=Coordinates(x=170.0, y=0.0, z=0.0),
            camera_rotation_deg=Coordinates(x=0.0, y=0.0, z=13.1),
            monitor_rotation_deg=Coordinates(x=0.0, y=0.0, z=0.0),
            equipment='CAM2P.1'
        )
        assert rg == expected

    def test_only_single_geometry_returned(self):
        """Tests that when a rig contains multiple geometries, that only 1 is
        returned"""
        dir = Path(__file__).parent.parent.resolve()
        test_data_dir = dir / 'test_data'

        # This example contains multiple geometries per config
        df = pd.read_pickle(
            str(test_data_dir / 'raw_eye_tracking_rig_geometry.pkl'))

        obtained = RigGeometry._select_most_recent_geometry(rig_geometry=df)
        assert (obtained.groupby(obtained.index).size() == 1).all()


class TestFromJson(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.ophys_experiment_id = 994278291

        dir = Path(__file__).parent.parent.resolve()
        test_data_dir = dir / 'test_data'

        with open(test_data_dir / 'eye_tracking_rig_geometry.json') as f:
            x = json.load(f)
            x = x['rig_geometry']
            x = {'eye_tracking_rig_geometry': x}
            cls.expected = RigGeometry.from_json(dict_repr=x)

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
            x = {'eye_tracking_rig_geometry': x}
            cls.rig_geometry = RigGeometry.from_json(dict_repr=x)

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

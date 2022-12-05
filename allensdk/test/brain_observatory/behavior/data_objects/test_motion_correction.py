import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files\
    .rigid_motion_transform_file import \
    RigidMotionTransformFile
from allensdk.brain_observatory.behavior.data_objects.motion_correction \
    import \
    MotionCorrection
from allensdk.brain_observatory.behavior.data_objects.timestamps\
    .ophys_timestamps import \
    OphysTimestamps
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest
from allensdk.test.brain_observatory.behavior.data_objects.metadata\
    .test_behavior_ophys_metadata import \
    TestBOM
from allensdk.test.brain_observatory.behavior.data_objects.test_cell_specimens\
    import TestJson


class TestFromDataFile(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.ophys_experiment_id = 994278291

    @pytest.mark.requires_bamboo
    def test_from_data_file(self):
        motion_correction_file = RigidMotionTransformFile.from_lims(
            ophys_experiment_id=self.ophys_experiment_id, db=self.dbconn)
        mc = MotionCorrection.from_data_file(
            rigid_motion_transform_file=motion_correction_file)
        assert not mc.value.empty
        expected_cols = ['x', 'y']
        assert len(mc.value.columns) == 2
        for c in expected_cols:
            assert c in mc.value.columns

class TestNWB:
    @classmethod
    def setup_class(cls):
        df = pd.DataFrame({'x': [2, 3, 2], 'y': [-3, -4, -4]})
        cls.motion_correction = MotionCorrection(motion_correction=df)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

        def _write_cell_specimen():
            # write metadata
            tbom = TestBOM()
            tbom.setup_class()
            bom = tbom.meta
            bom.to_nwb(nwbfile=self.nwbfile)

            # write cell specimen
            tjson = TestJson()
            tjson.setup_class()
            ophys_timestamps = OphysTimestamps(
                timestamps=np.array([.1, .2, .3]))
            tjson.csp.to_nwb(nwbfile=self.nwbfile, ophys_timestamps=ophys_timestamps)

        # need to write cell specimen, since it is a dependency
        _write_cell_specimen()

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self.motion_correction.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=MotionCorrection)
        else:
            obt = self.motion_correction.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.motion_correction

import datetime
import json

import pytest
from pynwb import NWBFile

from allensdk.brain_observatory.ecephys.probes import Probes
from allensdk.brain_observatory.ecephys.write_nwb.schemas import Probe


class TestProbes:
    @classmethod
    def setup_class(cls):
        with open('/allen/aibs/informatics/module_test_data/ecephys/'
                  'BEHAVIOR_ECEPHYS_WRITE_NWB_QUEUE_1111216934_input.json') \
                as f:
            input_data = json.load(f)

        # trim down the number of probes to reduce memory footprint of test
        input_data['session_data']['probes'] = (
                input_data['session_data']['probes'][:3])

        cls.input_data = input_data['session_data']['probes']
        probes = Probe().load(cls.input_data, many=True)
        cls._probes_from_json = Probes.from_json(probes=probes)

    def setup_method(self, method):
        self._nwbfile = NWBFile(
            session_description='foo',
            identifier='foo',
            session_id='foo',
            session_start_time=datetime.datetime.now(),
            institution="Allen Institute"
        )

    @pytest.mark.requires_bamboo
    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self._probes_from_json.to_nwb(nwbfile=self._nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self._nwbfile,
                data_object_cls=Probes)
        else:
            obt = Probes.from_nwb(nwbfile=self._nwbfile)

        assert obt == self._probes_from_json

    @pytest.mark.requires_bamboo
    def test_skip_probes(self):
        """tests that when skip_probes is passed that the probe is skipped"""
        names = [p['name'] for p in self.input_data]
        skip_probes = [names[0]]
        probes = Probes.from_json(
            probes=self.input_data, skip_probes=skip_probes)
        assert sorted([p.name for p in probes]) == \
               sorted([p for p in names if p not in skip_probes])

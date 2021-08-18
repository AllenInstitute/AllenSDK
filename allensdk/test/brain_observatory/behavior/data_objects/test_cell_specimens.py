import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pynwb

import pytest

from allensdk.brain_observatory.behavior.data_objects.cell_specimens.\
    cell_specimens import CellSpecimens, CellSpecimenMeta
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.imaging_plane import \
    ImagingPlane
from allensdk.brain_observatory.behavior.data_objects.timestamps\
    .ophys_timestamps import \
    OphysTimestamps
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.internal.api import db_connection_creator
from allensdk.test.brain_observatory.behavior.data_objects.metadata\
    .test_behavior_ophys_metadata import \
    TestBOM


class TestLims:
    @classmethod
    def setup_class(cls):
        cls.ophys_experiment_id = 994278291
        cls.expected_meta = CellSpecimenMeta(
            emission_lambda=520.0,
            imaging_plane=ImagingPlane(
                excitation_lambda=910.0,
                indicator='GCaMP6f',
                ophys_frame_rate=10.0,
                targeted_structure='VISp'
            )
        )

    def setup_method(self, method):
        marks = getattr(method, 'pytestmark', None)
        if marks:
            marks = [m.name for m in marks]

            # Will only create a dbconn if the test requires_bamboo
            if 'requires_bamboo' in marks:
                self.dbconn = db_connection_creator(
                    fallback_credentials=LIMS_DB_CREDENTIAL_MAP)

    @pytest.mark.requires_bamboo
    def test_from_internal(self):
        number_of_frames = 140296
        ots = OphysTimestamps(timestamps=np.linspace(start=.1,
                                                     stop=.1*number_of_frames,
                                                     num=number_of_frames),
                              number_of_frames=number_of_frames)
        csp = CellSpecimens.from_lims(
            ophys_experiment_id=self.ophys_experiment_id, lims_db=self.dbconn,
            ophys_timestamps=ots)
        assert not csp.table.empty
        assert not csp.events.empty
        assert not csp.dff_traces.empty
        assert not csp.corrected_fluorescence_traces.empty
        assert csp.meta == self.expected_meta


class TestJson:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'
        with open(test_data_dir / 'test_input.json') as f:
            dict_repr = json.load(f)
        dict_repr = dict_repr['session_data']
        dict_repr['sync_file'] = str(test_data_dir / 'sync.h5')
        dict_repr['behavior_stimulus_file'] = str(test_data_dir /
                                                  'behavior_stimulus_file.pkl')
        dict_repr['dff_file'] = str(test_data_dir / 'demix_file.h5')
        dict_repr['demix_file'] = str(test_data_dir / 'demix_file.h5')
        dict_repr['events_file'] = str(test_data_dir / 'events.h5')

        cls.dict_repr = dict_repr
        cls.expected_meta = CellSpecimenMeta(
            emission_lambda=520.0,
            imaging_plane=ImagingPlane(
                excitation_lambda=910.0,
                indicator='GCaMP6f',
                ophys_frame_rate=10.0,
                targeted_structure='VISp'
            )
        )
        cls.ophys_timestamps = OphysTimestamps(
            timestamps=np.array([.1, .2, .3]), number_of_frames=3)

    def test_from_json(self):
        csp = CellSpecimens.from_json(dict_repr=self.dict_repr,
                                      ophys_timestamps=self.ophys_timestamps)
        assert not csp.table.empty
        assert not csp.events.empty
        assert not csp.dff_traces.empty
        assert not csp.corrected_fluorescence_traces.empty
        assert csp.meta == self.expected_meta


class TestNWB:
    @classmethod
    def setup_class(cls):
        cls.ophys_timestamps = OphysTimestamps(
            timestamps=np.array([.1, .2, .3]), number_of_frames=3)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

        tj = TestJson()
        tj.setup_class()
        self.cell_specimens = CellSpecimens.from_json(
            dict_repr=tj.dict_repr, ophys_timestamps=self.ophys_timestamps)

        # Write metadata, since csp requires other metdata
        tbom = TestBOM()
        tbom.setup_class()
        bom = tbom.meta
        bom.to_nwb(nwbfile=self.nwbfile)

    @pytest.mark.parametrize('filter_invalid_rois', [True, False])
    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture,
                            filter_invalid_rois):
        csp = self.cell_specimens._cell_specimen_table

        if filter_invalid_rois:
            # changing one of the rois to be valid
            csp.loc[1086633332, 'valid_roi'] = True

        valid_roi_id = csp[csp['valid_roi']]['cell_roi_id']

        self.cell_specimens.to_nwb(nwbfile=self.nwbfile,
                                   ophys_timestamps=self.ophys_timestamps)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=CellSpecimens,
                filter_invalid_rois=filter_invalid_rois)
        else:
            obt = self.cell_specimens.from_nwb(
                nwbfile=self.nwbfile, filter_invalid_rois=filter_invalid_rois)

        if filter_invalid_rois:
            self.cell_specimens._cell_specimen_table = \
                self.cell_specimens._cell_specimen_table[
                    self.cell_specimens._cell_specimen_table['cell_roi_id']
                        .isin(valid_roi_id)]

        assert obt == self.cell_specimens

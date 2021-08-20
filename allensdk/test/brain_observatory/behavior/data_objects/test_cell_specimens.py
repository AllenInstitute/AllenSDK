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
            ophys_timestamps=ots,
            segmentation_mask_image_spacing=(.78125e-3, .78125e-3))
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
        csp = CellSpecimens.from_json(
            dict_repr=self.dict_repr,
            ophys_timestamps=self.ophys_timestamps,
            segmentation_mask_image_spacing=(.78125e-3, .78125e-3))
        assert not csp.table.empty
        assert not csp.events.empty
        assert not csp.dff_traces.empty
        assert not csp.corrected_fluorescence_traces.empty
        assert csp.meta == self.expected_meta

    @pytest.mark.parametrize('trace_type',
                             ('dff_traces',
                              'corrected_fluorescence_traces'))
    def test_traces_same_order_as_cell_specimen_table(self, trace_type):
        """tests that traces are in same order as cell specimen table"""
        csp = CellSpecimens.from_json(
            dict_repr=self.dict_repr,
            ophys_timestamps=self.ophys_timestamps,
            segmentation_mask_image_spacing=(.78125e-3, .78125e-3))
        private_trace_attr = getattr(csp, f'_{trace_type}')
        public_trace_attr = getattr(csp, trace_type)

        current_order = np.where(
            private_trace_attr.value.index ==
            csp._cell_specimen_table['cell_roi_id'])[0]

        # make sure same order
        private_trace_attr._value = private_trace_attr.value.iloc[current_order]

        # rearrange
        private_trace_attr._value = private_trace_attr._value.iloc[[2, 0, 1]]

        # make sure same order
        np.testing.assert_array_equal(public_trace_attr.index, csp.table.index)

    @pytest.mark.parametrize('extra_in_trace', (True, False))
    @pytest.mark.parametrize('trace_type',
                             ('dff_traces',
                              'corrected_fluorescence_traces'))
    def test_trace_rois_different_than_cell_specimen_table(self, trace_type,
                                                           extra_in_trace):
        """check that an exception is raised if there is a mismatch in rois
        between cell specimen table and traces"""
        csp = CellSpecimens.from_json(
            dict_repr=self.dict_repr,
            ophys_timestamps=self.ophys_timestamps,
            segmentation_mask_image_spacing=(.78125e-3, .78125e-3))
        private_trace_attr = getattr(csp, f'_{trace_type}')

        if extra_in_trace:
            # Drop an roi from cell specimen table that is in trace
            trace_rois = private_trace_attr.value.index
            csp._cell_specimen_table = csp._cell_specimen_table[
                csp._cell_specimen_table['cell_roi_id'] != trace_rois[0]]
        else:
            # Drop an roi from trace that is in cell specimen table
            csp_rois = csp._cell_specimen_table['cell_roi_id']
            private_trace_attr._value = private_trace_attr._value[
                private_trace_attr._value.index != csp_rois.iloc[0]]

        if trace_type == 'dff_traces':
            trace_args = {
                'dff_traces': private_trace_attr,
                'corrected_fluorescence_traces':
                    csp._corrected_fluorescence_traces
            }
        else:
            trace_args = {
                'dff_traces': csp._dff_traces,
                'corrected_fluorescence_traces': private_trace_attr
            }
        with pytest.raises(RuntimeError):
            # construct it again using trace/table combo with different rois
            CellSpecimens(
                cell_specimen_table=csp._cell_specimen_table,
                meta=csp._meta,
                events=csp._events,
                ophys_timestamps=self.ophys_timestamps,
                segmentation_mask_image_spacing=(.78125e-3, .78125e-3),
                **trace_args
            )


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
        self.dict_repr = tj.dict_repr

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
        if filter_invalid_rois:
            # changing one of the rois to be valid
            self.dict_repr['cell_specimen_table_dict']['valid_roi']['0'] = True

        cell_specimens = CellSpecimens.from_json(
            dict_repr=self.dict_repr, ophys_timestamps=self.ophys_timestamps,
            segmentation_mask_image_spacing=(.78125e-3, .78125e-3),
            filter_invalid_rois=filter_invalid_rois)

        csp = cell_specimens._cell_specimen_table

        valid_roi_id = csp[csp['valid_roi']]['cell_roi_id']

        cell_specimens.to_nwb(nwbfile=self.nwbfile,
                              ophys_timestamps=self.ophys_timestamps)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=CellSpecimens,
                filter_invalid_rois=filter_invalid_rois,
                segmentation_mask_image_spacing=(.78125e-3, .78125e-3))
        else:
            obt = cell_specimens.from_nwb(
                nwbfile=self.nwbfile, filter_invalid_rois=filter_invalid_rois,
                segmentation_mask_image_spacing=(.78125e-3, .78125e-3))

        if filter_invalid_rois:
            cell_specimens._cell_specimen_table = \
                cell_specimens._cell_specimen_table[
                    cell_specimens._cell_specimen_table['cell_roi_id']
                    .isin(valid_roi_id)]

        assert obt == cell_specimens

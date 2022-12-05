import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pynwb
import pandas as pd

import pytest

from allensdk.core import DataObject
from allensdk.brain_observatory.behavior.data_files.neuropil_corrected_file \
    import NeuropilCorrectedFile
from allensdk.brain_observatory.behavior.data_files.demix_file import DemixFile
from allensdk.brain_observatory.behavior.data_files.neuropil_file import (
    NeuropilFile,
)
from allensdk.brain_observatory.behavior.data_files.dff_file import DFFFile
from allensdk.brain_observatory.behavior.data_files.event_detection_file \
    import EventDetectionFile
from allensdk.brain_observatory.behavior.data_objects.cell_specimens.traces \
    .dff_traces import \
    DFFTraces
from allensdk.brain_observatory.behavior.data_objects.cell_specimens.traces\
    .corrected_fluorescence_traces import CorrectedFluorescenceTraces
from allensdk.brain_observatory.behavior.data_objects.cell_specimens.traces\
    .demixed_traces import DemixedTraces
from allensdk.brain_observatory.behavior.data_objects.cell_specimens.traces\
    .neuropil_traces import NeuropilTraces
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.field_of_view_shape import FieldOfViewShape
from allensdk.brain_observatory.behavior.data_objects.cell_specimens\
    .cell_specimens import (
        CellSpecimens,
        CellSpecimenMeta,
        EventsParams,
    )
from allensdk.brain_observatory.behavior.data_objects.cell_specimens\
    .rois_mixin import RoisMixin
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.imaging_plane import ImagingPlane
from allensdk.brain_observatory.behavior.data_objects.timestamps\
    .ophys_timestamps import OphysTimestamps
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.internal.api import db_connection_creator
from allensdk.test.brain_observatory.behavior.data_objects.metadata\
    .test_behavior_ophys_metadata import (
        TestBOM,
    )


class TestLims:
    @classmethod
    def setup_class(cls):
        cls.ophys_experiment_id = 994278291
        cls.expected_meta = CellSpecimenMeta(
            emission_lambda=520.0,
            imaging_plane=ImagingPlane(
                excitation_lambda=910.0,
                indicator="GCaMP6f",
                ophys_frame_rate=10.0,
                targeted_structure="VISp",
            ),
        )

    def setup_method(self, method):
        marks = getattr(method, "pytestmark", None)
        if marks:
            marks = [m.name for m in marks]

            # Will only create a dbconn if the test requires_bamboo
            if "requires_bamboo" in marks:
                self.dbconn = db_connection_creator(
                    fallback_credentials=LIMS_DB_CREDENTIAL_MAP
                )

    @pytest.mark.requires_bamboo
    def test_from_lims(self):
        number_of_frames = 140296
        ots = OphysTimestamps(
            timestamps=np.linspace(
                start=0.1, stop=0.1 * number_of_frames, num=number_of_frames
            )
        )
        csp = CellSpecimens.from_lims(
            ophys_experiment_id=self.ophys_experiment_id,
            lims_db=self.dbconn,
            ophys_timestamps=ots,
            segmentation_mask_image_spacing=(0.78125e-3, 0.78125e-3),
            events_params=EventsParams(
                filter_scale_seconds=2.0 / 31.0, filter_n_time_steps=20
            ),
        )
        assert not csp.table.empty
        assert not csp.events.empty
        assert not csp.dff_traces.empty
        assert not csp.corrected_fluorescence_traces.empty
        assert csp.meta == self.expected_meta


class TestJson:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / "test_data"
        with open(test_data_dir / "test_input.json") as f:
            dict_repr = json.load(f)
        dict_repr = dict_repr["session_data"]
        dict_repr["sync_file"] = str(test_data_dir / "sync.h5")
        dict_repr["behavior_stimulus_file"] = str(
            test_data_dir / "behavior_stimulus_file.pkl"
        )
        dict_repr["dff_file"] = str(test_data_dir / "demix_file.h5")
        dict_repr["demix_file"] = str(test_data_dir / "demix_file.h5")
        dict_repr["neuropil_file"] = str(test_data_dir / "demix_file.h5")
        dict_repr["neuropil_corrected_file"] = str(
            test_data_dir / "neuropil_corrected_file.h5"
        )
        dict_repr["events_file"] = str(test_data_dir / "events.h5")

        cls.dict_repr = dict_repr
        cls.expected_meta = CellSpecimenMeta(
            emission_lambda=520.0,
            imaging_plane=ImagingPlane(
                excitation_lambda=910.0,
                indicator="GCaMP6f",
                ophys_frame_rate=10.0,
                targeted_structure="VISp",
            ),
        )
        cls.ophys_timestamps = OphysTimestamps(
            timestamps=np.array([0.1, 0.2, 0.3])
        )
        cell_specimen_table = CellSpecimens._postprocess(
            cell_specimen_table=dict_repr['cell_specimen_table_dict'],
            fov_shape=FieldOfViewShape(
                height=dict_repr['movie_height'], width=dict_repr['movie_width']
                )
            )

        def _get_dff_traces():
            dff_file = DFFFile(filepath=dict_repr['dff_file'])
            return DFFTraces.from_data_file(
                dff_file=dff_file)

        def _get_demixed_traces():
            demix_file = DemixFile(filepath=dict_repr['demix_file'])
            return DemixedTraces.from_data_file(
                demix_file=demix_file
            )

        def _get_neuropil_traces():
            neuropil_file = NeuropilFile(filepath=dict_repr['neuropil_file'])
            return NeuropilTraces.from_data_file(
                neuropil_file=neuropil_file
            )

        def _get_corrected_fluorescence_traces():
            neuropil_corrected_file = NeuropilCorrectedFile(
                filepath=dict_repr['neuropil_corrected_file'])
            return CorrectedFluorescenceTraces.from_data_file(
                neuropil_corrected_file=neuropil_corrected_file
            )
        
        def _get_events():
            events_file = EventDetectionFile(
                filepath=dict_repr['events_file']),
            return CellSpecimens._get_events(
                events_file=events_file[0],
                events_params=EventsParams(
                    filter_scale_seconds=2.0/31.0,
                    filter_n_time_steps=20),
                    frame_rate_hz=cls.expected_meta.\
                        imaging_plane.ophys_frame_rate,
            )
        dff_traces = _get_dff_traces()
        demixed_traces = _get_demixed_traces()
        neuropil_traces = _get_neuropil_traces()
        corrected_fluorescence_traces = _get_corrected_fluorescence_traces()
        events = _get_events()
        cls.csp = CellSpecimens(
            cell_specimen_table=cell_specimen_table, meta=cls.expected_meta,
            dff_traces=dff_traces,
            demixed_traces=demixed_traces,
            neuropil_traces=neuropil_traces,
            corrected_fluorescence_traces=corrected_fluorescence_traces,
            events=events,
            ophys_timestamps=cls.ophys_timestamps,
            segmentation_mask_image_spacing=(0.78125e-3, 0.78125e-3),
            )

    @pytest.mark.parametrize(
        "data",
        (
            "dff_traces",
            "demixed_traces",
            "neuropil_traces",
            "corrected_fluorescence_traces",
            "events",
        ),
    )
    def test_roi_data_same_order_as_cell_specimen_table(self, data):
        """tests that roi data are in same order as cell specimen table"""
        private_attr = getattr(self.csp, f"_{data}")
        public_attr = getattr(self.csp, data)

        # Events stores cell_roi_id as column whereas traces is index
        data_cell_roi_ids = getattr(
            private_attr.value, "cell_roi_id" if data == "events" else "index"
        ).values

        current_order = np.where(
            data_cell_roi_ids == self.csp._cell_specimen_table["cell_roi_id"]
        )[0]

        # make sure same order
        private_attr._value = private_attr.value.iloc[current_order]

        # rearrange
        private_attr._value = private_attr._value.iloc[[1, 0]]

        # make sure same order
        np.testing.assert_array_equal(public_attr.index, self.csp.table.index)

    @pytest.mark.parametrize("extra_in_trace", (True, False))
    @pytest.mark.parametrize(
        "trace_type",
        (
            "dff_traces",
            "demixed_traces",
            "neuropil_traces",
            "corrected_fluorescence_traces",
        ),
    )
    def test_trace_rois_different_than_cell_specimen_table(
        self, trace_type, extra_in_trace
    ):
        """check that an exception is raised if there is a mismatch in rois
        between cell specimen table and traces"""
        private_trace_attr = getattr(self.csp, f"_{trace_type}")

        if extra_in_trace:
            # Drop an roi from cell specimen table that is in trace
            trace_rois = private_trace_attr.value.index
            self.csp._cell_specimen_table = self.csp._cell_specimen_table[
                self.csp._cell_specimen_table["cell_roi_id"] != trace_rois[0]
            ]
        else:
            # Drop an roi from trace that is in cell specimen table
            csp_rois = self.csp._cell_specimen_table["cell_roi_id"]
            private_trace_attr._value = private_trace_attr._value[
                private_trace_attr._value.index != csp_rois.iloc[0]
            ]

        if trace_type == "dff_traces":
            trace_args = {
                "dff_traces": private_trace_attr,
                "demixed_traces": self.csp._demixed_traces,
                "neuropil_traces": self.csp._neuropil_traces,
                "corrected_fluorescence_traces":
                    self.csp._corrected_fluorescence_traces,
            }
        elif trace_type == "demixed_traces":
            trace_args = {
                "dff_traces": self.csp._dff_traces,
                "demixed_traces": private_trace_attr,
                "neuropil_traces": self.csp._neuropil_traces,
                "corrected_fluorescence_traces":
                    self.csp._corrected_fluorescence_traces,
            }
        elif trace_type == "neuropil_traces":
            trace_args = {
                "dff_traces": self.csp._dff_traces,
                "demixed_traces": self.csp._demixed_traces,
                "neuropil_traces": private_trace_attr,
                "corrected_fluorescence_traces":
                    self.csp._corrected_fluorescence_traces,
            }
        else:
            trace_args = {
                "dff_traces": self.csp._dff_traces,
                "demixed_traces": self.csp._demixed_traces,
                "neuropil_traces": self.csp._neuropil_traces,
                "corrected_fluorescence_traces": private_trace_attr,
            }
        with pytest.raises(RuntimeError):
            # construct it again using trace/table combo with different rois
            CellSpecimens(
                cell_specimen_table=self.csp._cell_specimen_table,
                meta=self.csp._meta,
                events=self.csp._events,
                ophys_timestamps=self.ophys_timestamps,
                segmentation_mask_image_spacing=(0.78125e-3, 0.78125e-3),
                exclude_invalid_rois=False,
                **trace_args,
            )


class TestNWB:
    @classmethod
    def setup_class(cls):
        cls.ophys_timestamps = OphysTimestamps(
            timestamps=np.array([0.1, 0.2, 0.3])
        )

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description="asession",
            identifier="1234",
            session_start_time=datetime.now(),
        )

        tj = TestJson()
        tj.setup_class()
        self.csp = tj.csp
        # Write metadata, since csp requires other metdata
        tbom = TestBOM()
        tbom.setup_class()
        bom = tbom.meta
        bom.to_nwb(nwbfile=self.nwbfile)

    @pytest.mark.parametrize("exclude_invalid_rois", [True, False])
    @pytest.mark.parametrize("roundtrip", [True, False])
    def test_read_write_nwb(
        self, roundtrip, data_object_roundtrip_fixture, exclude_invalid_rois
    ):
        cell_specimen_table = self.csp._cell_specimen_table
        valid_roi_id = cell_specimen_table[cell_specimen_table["valid_roi"]]["cell_roi_id"]

        self.csp.to_nwb(
            nwbfile=self.nwbfile, ophys_timestamps=self.ophys_timestamps
        )

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=CellSpecimens,
                exclude_invalid_rois=exclude_invalid_rois,
                segmentation_mask_image_spacing=(0.78125e-3, 0.78125e-3),
                events_params=EventsParams(
                    filter_scale_seconds=2.0 / 31.0, filter_n_time_steps=20
                ),
            )
        else:
            obt = self.csp.from_nwb(
                nwbfile=self.nwbfile,
                exclude_invalid_rois=exclude_invalid_rois,
                segmentation_mask_image_spacing=(0.78125e-3, 0.78125e-3),
                events_params=EventsParams(
                    filter_scale_seconds=2.0 / 31.0, filter_n_time_steps=20
                ),
            )

        if exclude_invalid_rois:
            self.csp._cell_specimen_table = (
                self.csp._cell_specimen_table[
                    self.csp._cell_specimen_table["cell_roi_id"].isin(
                        valid_roi_id
                    )
                ]
            )

        assert obt == self.csp


class TestFilterAndReorder:
    @pytest.mark.parametrize("raise_if_rois_missing", (True, False))
    def test_missing_rois(self, raise_if_rois_missing):
        """Tests that when dataframe missing rois, that they are ignored"""
        roi_ids = np.array([1, 2])
        df = pd.DataFrame({"cell_roi_id": [1], "foo": [2]})

        class Rois(DataObject, RoisMixin):
            def __init__(self):
                super().__init__(name="test", value=df)

        rois = Rois()
        if raise_if_rois_missing:
            with pytest.raises(RuntimeError):
                rois.filter_and_reorder(
                    roi_ids=roi_ids,
                    raise_if_rois_missing=raise_if_rois_missing,
                )
        else:
            rois.filter_and_reorder(
                roi_ids=roi_ids, raise_if_rois_missing=raise_if_rois_missing
            )
            expected = pd.DataFrame({"cell_roi_id": [1], "foo": [2]})
            pd.testing.assert_frame_equal(rois._value, expected)

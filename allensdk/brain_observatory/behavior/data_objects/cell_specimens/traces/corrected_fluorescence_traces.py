import numpy as np
import pandas as pd
from pynwb import NWBFile
from pynwb.ophys import Fluorescence

from allensdk.brain_observatory.behavior.data_files.demix_file import DemixFile
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    DataFileReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.cell_specimens\
    .rois_mixin import \
    RoisMixin


class CorrectedFluorescenceTraces(DataObject, RoisMixin,
                                  DataFileReadableInterface,
                                  NwbReadableInterface, NwbWritableInterface):
    def __init__(self, traces: pd.DataFrame):
        """

        Parameters
        ----------
        traces
            index cell_roi_id
            columns:
            - corrected_fluorescence
                list of float
        """
        super().__init__(name='corrected_fluorescence_traces', value=traces)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) \
            -> "CorrectedFluorescenceTraces":
        corr_fluorescence_nwb = nwbfile.processing[
            'ophys'].data_interfaces[
            'corrected_fluorescence'].roi_response_series['traces']
        # f traces stored as timepoints x rois in NWB
        # We want rois x timepoints, hence the transpose
        f_traces = corr_fluorescence_nwb.data[:].T
        df = pd.DataFrame({'corrected_fluorescence': f_traces.tolist()},
                          index=pd.Index(
                              data=corr_fluorescence_nwb.rois.table.id[:],
                              name='cell_roi_id'))
        return cls(traces=df)

    @classmethod
    def from_data_file(cls,
                       demix_file: DemixFile) \
            -> "CorrectedFluorescenceTraces":
        corrected_fluorescence_traces = demix_file.data
        return cls(traces=corrected_fluorescence_traces)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        corrected_fluorescence_traces = self.value['corrected_fluorescence']

        # Convert from Series of lists to numpy array
        # of shape ROIs x timepoints
        traces = np.stack(
            [x for x in corrected_fluorescence_traces])

        # Create/Add corrected_fluorescence_traces modules and interfaces:
        ophys_module = nwbfile.processing['ophys']

        roi_table_region = \
            nwbfile.processing['ophys'].data_interfaces[
                'dff'].roi_response_series[
                'traces'].rois  # noqa: E501
        ophys_timestamps = ophys_module.get_data_interface(
            'dff').roi_response_series['traces'].timestamps
        f_interface = Fluorescence(name='corrected_fluorescence')
        ophys_module.add_data_interface(f_interface)

        f_interface.create_roi_response_series(
            name='traces',
            data=traces.T,  # Should be stored as timepoints x rois
            unit='NA',
            rois=roi_table_region,
            timestamps=ophys_timestamps)
        return nwbfile

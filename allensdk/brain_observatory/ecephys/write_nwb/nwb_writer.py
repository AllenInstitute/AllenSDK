import inspect
import os
from pathlib import Path
from typing import Union

from pynwb import NWBHDF5IO, NWBFile

from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession
from allensdk.brain_observatory.nwb.nwb_utils import NWBWriter
from allensdk.core import JsonReadableInterface, NwbReadableInterface, \
    NwbWritableInterface


class BehaviorEcephysNwbWriter(NWBWriter):
    """NWB Writer for behavior ecephys. Same as `NWBWriter` except also
    writes probe NWB files separately """
    def __init__(
            self,
            session_nwb_filepath: str,
            session_data: dict,
            serializer: Union[
                JsonReadableInterface,
                NwbReadableInterface,
                NwbWritableInterface]):
        super().__init__(
            nwb_filepath=session_nwb_filepath,
            session_data=session_data,
            serializer=serializer
        )

    def _write_nwb(
            self,
            session: BehaviorEcephysSession,
            **kwargs) -> NWBFile:
        to_nwb_kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(self._serializer.to_nwb).parameters}
        session_nwbfile, probe_nwbfile_map = session.to_nwb(**to_nwb_kwargs)

        os.makedirs(Path(self.nwb_filepath_inprogress).parent, exist_ok=True)

        with NWBHDF5IO(self.nwb_filepath_inprogress, 'w') as nwb_file_writer:
            nwb_file_writer.write(session_nwbfile)

        for probe_name, probe_nwbfile in probe_nwbfile_map.items():
            probe_id = [p.id for p in session.probes
                        if p.name == probe_name][0]
            if probe_nwbfile is not None:
                probe_nwb_path = Path(self._nwb_filepath).parent / \
                    f'lfp_probe_{probe_id}.nwb'
                with NWBHDF5IO(probe_nwb_path, 'w') as nwb_file_writer:
                    nwb_file_writer.write(probe_nwbfile)
        return session_nwbfile

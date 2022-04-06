from typing import List, Optional

import numpy as np
import pandas as pd
from pynwb import NWBFile

from allensdk.brain_observatory.ecephys._channel import Channel
from allensdk.brain_observatory.ecephys.utils import clobbering_merge
from allensdk.core import DataObject, NwbReadableInterface, \
    JsonReadableInterface


class Channels(DataObject, NwbReadableInterface, JsonReadableInterface):
    """A set of channels"""

    def __init__(self, channels: List[Channel]):
        super().__init__(name='channels', value=channels)

    @classmethod
    def from_json(cls, channels: dict) -> "Channels":
        channels = [Channel(**{f'{"impedance" if k == "impedence" else k}': v
                               for k, v in channel.items()})
                    for channel in channels]
        return Channels(channels=channels)

    def to_dataframe(self, external_channel_columns=None,
                     filter_by_validity=True) -> pd.DataFrame:
        """

        Parameters
        ----------
        external_channel_columns
        filter_by_validity: Whether to filter channels based on whether
            the channel is marked as "valid_data"

        Returns
        -------

        """
        channels = [channel.to_dict()['channel'] for channel in self.value]
        channels = pd.DataFrame(channels)
        channels = channels.set_index('id')
        channels = channels.drop(columns=['impedance'])

        if external_channel_columns is not None:
            external_channel_columns = external_channel_columns()
            channels = clobbering_merge(channels, external_channel_columns,
                                        left_index=True, right_index=True)

        if filter_by_validity:
            channels = channels[channels['valid_data']]
            channels = channels.drop(columns=['valid_data'])
        return channels

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile,
                 probe_id: Optional[int] = None) -> "Channels":
        """

        Parameters
        ----------
        nwbfile
        probe_id: Filter channels to only those for `probe_id`

        Returns
        -------
        Channels
        """
        channels = []
        for channel_id, row in nwbfile.electrodes.to_dataframe().iterrows():
            if probe_id is not None:
                if row['probe_id'] != probe_id:
                    continue
            manual_structure_acronym = \
                np.nan if row['location'] in ['None', ''] else row['location']
            channels.append(Channel(
                id=channel_id,
                local_index=row['local_index'],
                probe_horizontal_position=row['probe_horizontal_position'],
                probe_vertical_position=row['probe_vertical_position'],
                probe_id=row['probe_id'],
                valid_data=row['valid_data'],
                manual_structure_acronym=manual_structure_acronym,
                manual_structure_id=STRUCTURE_ACRONYM_ID_MAP.get(
                    manual_structure_acronym, np.nan),
                anterior_posterior_ccf_coordinate=row['x'],
                dorsal_ventral_ccf_coordinate=row['y'],
                left_right_ccf_coordinate=row['z']
            ))
        return Channels(channels=channels)


STRUCTURE_ACRONYM_ID_MAP = {
    "grey": 8, "SCig": 10, "SCiw": 17, "IGL": 27, "LT": 66, "VL": 81,
    "MRN": 128, "LD": 155, "LGd": 170, "LGv": 178, "APN": 215, "LP": 218,
    "RT": 262, "MB": 313, "SGN": 325, "BMAa": 327, "CA": 375, "CA1": 382,
    "VISp": 385, "VISam": 394, "VISal": 402, "VISl": 409, "VISrl": 417,
    "CA2": 423, "CA3": 463, "SUB": 502, "VISpm": 533, "TH": 549,
    "NOT": 628, "COAa": 639, "COApm": 663, "VIS": 669, "CP": 672,
    "OLF": 698, "OP": 706, "VPL": 718, "DG": 726, "VPM": 733, "ZI": 797,
    "SCzo": 834, "SCsg": 842, "SCop": 851, "PF": 930, "PO": 1020,
    "POL": 1029, "POST": 1037, "PP": 1044, "PPT": 1061, "MGd": 1072,
    "MGv": 1079, "PRE": 1084, "MGm": 1088, "HPF": 1089,
    "VISli": 312782574, "VISmma": 480149258, "VISmmp": 480149286,
    "ProS": 484682470, "RPF": 549009203, "Eth": 560581551,
    "PIL": 560581563, "PoT": 563807435, "IntG": 563807439
}

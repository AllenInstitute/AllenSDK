import pandas as pd
import pynwb
import SimpleITK as sitk
import os
import collections

from allensdk.brain_observatory.running_speed import RunningSpeed
from allensdk.brain_observatory.behavior.image_api import ImageApi

from pynwb import load_namespaces
namespace_path = os.path.join(os.path.dirname(__file__), 'AIBS_ophys_behavior_namespace.yaml')
load_namespaces(namespace_path)


class NwbApi:

    __slots__ = ('path', '_nwbfile')

    @property
    def nwbfile(self):
        if hasattr(self, '_nwbfile'):
            return self._nwbfile

        io = pynwb.NWBHDF5IO(self.path, 'r')
        return io.read()

    def __init__(self, path, **kwargs):
        ''' Reads data for a single Brain Observatory session from an NWB 2.0 file
        '''

        self.path = path

    @classmethod
    def from_nwbfile(cls, nwbfile, **kwargs):

        obj = cls(path=None, **kwargs)
        obj._nwbfile = nwbfile

        return obj

    @classmethod
    def from_path(cls, path, **kwargs):
        with open(path, 'r'):
            pass

        return cls(path=path, **kwargs)

    def get_running_speed(self) -> RunningSpeed:

        values = self.nwbfile.modules['running'].get_data_interface('speed').data[:]
        timestamps = self.nwbfile.modules['running'].get_data_interface('speed').timestamps[:]

        return RunningSpeed(
            timestamps=timestamps,
            values=values,
        )

    def get_stimulus_presentations(self) -> pd.DataFrame:

        columns_to_ignore = set(['tags', 'timeseries', 'tags_index', 'timeseries_index'])

        presentation_dfs = []
        for interval_name, interval in self.nwbfile.intervals.items():
            if interval_name.endswith('_presentations'):
                presentations = collections.defaultdict(list)
                for col in interval.columns:
                    if col.name not in columns_to_ignore:
                        presentations[col.name].extend(col.data)
                df = pd.DataFrame(presentations).replace({'N/A': ''})
                presentation_dfs.append(df)

        table = pd.concat(presentation_dfs, sort=False)
        table = table.sort_values(by=["start_time"])
        table = table.reset_index(drop=True)
        table.index.name = 'stimulus_presentations_id'
        table.index = table.index.astype(int)

        for colname, series in table.items():
            types = set(series.map(type))
            if len(types) > 1 and str in types:
                series.fillna('', inplace=True)
                table[colname] = series.transform(str)

        return table[sorted(table.columns)]

    def get_invalid_times(self) -> pd.DataFrame:

        container = self.nwbfile.invalid_times
        if container:
            return container.to_dataframe()
        else:
            return pd.DataFrame()

    def get_image(self, name, module, image_api=None) -> sitk.Image:

        if image_api is None:
            image_api = ImageApi

        nwb_img = self.nwbfile.modules[module].get_data_interface('images')[name]
        data = nwb_img.data
        resolution = nwb_img.resolution  # px/cm
        spacing = [resolution * 10, resolution * 10]

        return ImageApi.serialize(data, spacing, 'mm')

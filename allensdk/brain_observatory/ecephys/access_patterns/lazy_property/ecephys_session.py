from .ecephys_filesystem_api import EcephysFilesystemApi
from .lazy_property import LazyProperty


class EcephysSession:
    ''' Data for this session
    '''
    
    probes = LazyProperty('get_probes')
    channels = LazyProperty('get_channels')
    units = LazyProperty('get_units')
    mean_waveforms = LazyProperty('get_mean_waveforms')
    spike_times = LazyProperty('get_spike_times')
    running_speed = LazyProperty('get_running_speed')
    stimulus_table = LazyProperty('get_stimulus_table')
    
    def __init__(self, api):
        self.api = api
        
    @classmethod
    def from_path(cls, path, api_cls=EcephysFilesystemApi, **api_options):
        api = api_cls(path=path, **api_options)
        return cls(api=api)


class EcephysApi:

    __slots__: tuple = tuple([])

    def __init__(self, *args, **kwargs):
        pass

    def get_running_speed(self):
        raise NotImplementedError
    
    def get_stimulus_table(self):
        raise NotImplementedError
    
    def get_probes(self):
        raise NotImplementedError
    
    def get_channels(self):
        raise NotImplementedError
    
    def get_mean_waveforms(self):
        raise NotImplementedError

    def get_spike_times(self):
        raise NotImplementedError
    
    def get_units(self):
        raise NotImplementedError
class EphysDataSet( object ):
    def __init__(self, file_name):
        self.file_name = file_name
        
    def get_sweep(self, sweep_number):
        raise Exception("get_sweep not implemented")

    def set_sweep(self, sweep_number, stimulus, response):
        raise Exception("get_sweep not implemented")

    def get_spike_times(self, sweep_number):
        raise Exception("set_sweep_spike_times not implemented")

    def set_spike_times(self, sweep_number, spike_times):
        raise Exception("set_sweep_spike_times not implemented")


    

        
        

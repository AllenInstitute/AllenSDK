import h5py
from ephys_data_set import EphysDataSet

class OrcaDataSet( EphysDataSet ):
    def get_test_pulse(self, sweep_number):
        with h5py.File(self.file_name,'r') as f:
            sec = f['epochs']['TestPulse_%d' % sweep_number]
            stimulus = sec['stimulus']['timeseries']['data'].value
            response = sec['response']['timeseries']['data'].value
            
            return {
                'stimulus': stimulus,
                'response': response,
                'index_range': ( exp['stimulus']['idx_start'].value, exp['stimulus']['idx_stop'].value ),
                'sampling_rate': sec['stimulus']['timeseries']['sampling_rate'].value
            }

    def get_experiment(self, sweep_number):
        with h5py.File(self.file_name,'r') as f:
            sec = f['epochs']['Experiment_%d' % sweep_number]
            stimulus = sec['stimulus']['timeseries']['data'].value
            response = sec['response']['timeseries']['data'].value
            
            return {
                'stimulus': stimulus,
                'response': response,
                'index_range': ( exp['stimulus']['idx_start'].value, exp['stimulus']['idx_stop'].value ),
                'sampling_rate': sec['stimulus']['timeseries']['sampling_rate'].value
            }

    def get_full_sweep(self, sweep_number):
        with h5py.File(self.file_name,'r') as f:

            swp = f['epochs']['Sweep_%d' % sweep_number]
            
            stimulus = swp['stimulus']['timeseries']['data'].value
            response = swp['response']['timeseries']['data'].value
            
            out = {
                'stimulus': stimulus,
                'response': response,
                'sampling_rate': swp['stimulus']['timeseries']['sampling_rate'].value
            }
            
            try:
                # if the sweep has an experiment, the index range will point to the range of values containing the experimental data.
                exp = f['epochs']['Experiment_%d' % sweep_number]
                out['index_range'] = ( exp['stimulus']['idx_start'].value, exp['stimulus']['idx_stop'].value ),
            except KeyError, e:
                # this sweep has no experiment.  return the index range of the entire sweep.
                out['index_range'] = ( swp['stimulus']['idx_start'].value, swp['stimulus']['idx_stop'].value ),
                
            return out


    def set_full_sweep(self, sweep_number, stimulus, response):
        self.set_data('Sweep_%d' % sweep_number, stimulus, response)

    def set_experiment(self, sweep_number, stimulus, response):
        self.set_data('Experiment_%d' % sweep_number, stimulus, response)

    def set_test_pulse(self, sweep_number, stimulus, response):
        self.set_data('TestPulse_%d' % sweep_number, stimulus, response)

    def set_data(self, epoch, stimulus, response):
        with h5py.File(self.file_name,'r+') as f:
            ep = f['epochs'][epoch]

            if stimulus is not None:
                ep['stimulus']['timeseries']['data'][...] = stimulus

            if response is not None:
                ep['response']['timeseries']['data'][...] = response


import h5py
from ephys_data_set import EphysDataSet

class OrcaDataSet( EphysDataSet ):
    def get_test_pulse(self, sweep_number):
        with h5py.File(self.file_name,'r') as f:
            sec = f['epochs']['TestPulse_%d' % sweep_number]
            stimulus = sec['stimulus']['sequence']['data'].value
            response = sec['response']['sequence']['data'].value
            
            return {
                'stimulus': stimulus,
                'response': response,
                'index_range': (0, len(stimulus)-1),
                'dt': sec['stimulus']['sequence']['sampling_rate'].value
            }

    def get_experiment(self, sweep_number):
        with h5py.File(self.file_name,'r') as f:
            sec = f['epochs']['Experiment_%d' % sweep_number]
            stimulus = sec['stimulus']['sequence']['data'].value
            response = sec['response']['sequence']['data'].value
            
            return {
                'stimulus': stimulus,
                'response': response,
                'index_range': (0, len(stimulus)-1),
                'dt': sec['stimulus']['sequence']['sampling_rate'].value
            }

    def get_full_sweep(self, sweep_number):
        with h5py.File(self.file_name,'r') as f:
            exp = f['epochs']['Experiment_%d' % sweep_number]
            swp = f['epochs']['Sweep_%d' % sweep_number]
            
            stimulus = swp['stimulus']['sequence']['data'].value
            response = swp['response']['sequence']['data'].value
            
            return {
                'stimulus': stimulus,
                'response': response,
                'index_range': ( exp['stimulus']['idx_start'].value, exp['stimulus']['idx_stop'].value ),
                'dt': swp['stimulus']['sequence']['sampling_rate'].value
            }

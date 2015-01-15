import h5py
import numpy as np
from ephys_data_set import EphysDataSet

class OrcaDataSet( EphysDataSet ):

    def get_sweep(self, sweep_number):
        with h5py.File(self.file_name,'r') as f:

            swp = f['epochs']['Sweep_%d' % sweep_number]
            
            stimulus = swp['stimulus']['timeseries']['data'].value
            response = swp['response']['timeseries']['data'].value

            try:
                # if the sweep has an experiment, extract the experiment's index range
                exp = f['epochs']['Experiment_%d' % sweep_number]
                sweep_index_range = ( swp['stimulus']['idx_start'].value, swp['stimulus']['idx_stop'].value )
                experiment_index_range = ( exp['stimulus']['idx_start'].value, exp['stimulus']['idx_stop'].value )
            except KeyError, e:
                # this sweep has no experiment.  return the index range of the entire sweep.
                sweep_index_range = ( swp['stimulus']['idx_start'].value, swp['stimulus']['idx_stop'].value )
                experiment_index_range = sweep_index_range

            assert sweep_index_range[0] == 0, Exception("index range of the full sweep does not start at 0.")

            # only return data up to the end of the experiment -- ignore everything else
            return  {
                'stimulus': stimulus[sweep_index_range[0]:experiment_index_range[1]],
                'response': response[sweep_index_range[0]:experiment_index_range[1]],
                'index_range': experiment_index_range,
                'sampling_rate': swp['stimulus']['timeseries']['sampling_rate'].value
            }

            return out


    def set_sweep(self, sweep_number, stimulus, response):
        with h5py.File(self.file_name,'r+') as f:
            swp = f['epochs']['Sweep_%d' % sweep_number]

            sweep_index_range = ( swp['stimulus']['idx_start'].value, swp['stimulus']['idx_stop'].value )
            sweep_length = sweep_index_range[1] - sweep_index_range[0]
            
            if stimulus is not None:
                missing_data = sweep_length - len(stimulus)
                if missing_data > 0:
                    stimulus = np.append(stimulus, np.zeros(missing_data))

                swp['stimulus']['timeseries']['data'][...] = stimulus

            if response is not None:
                missing_data = sweep_length - len(response)
                if missing_data > 0:
                    response = np.append(response, np.zeros(missing_data))

                swp['response']['timeseries']['data'][...] = response


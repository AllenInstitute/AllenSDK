# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

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
                'stimulus': stimulus[sweep_index_range[0]:experiment_index_range[1]+1],
                'response': response[sweep_index_range[0]:experiment_index_range[1]+1],
                'index_range': experiment_index_range,
                'sampling_rate': swp['stimulus']['timeseries']['sampling_rate'].value
            }

            return out


    def set_sweep(self, sweep_number, stimulus, response):
        with h5py.File(self.file_name,'r+') as f:
            swp = f['epochs']['Sweep_%d' % sweep_number]

            # this is the length of the entire sweep data, including test pulse and whatever might be in front of it
            sweep_length = swp['stimulus']['idx_stop'].value + 1
            
            if stimulus is not None:
                # if the data is shorter than the sweep, pad it with zeros
                missing_data = sweep_length - len(stimulus)
                if missing_data > 0:
                    stimulus = np.append(stimulus, np.zeros(missing_data))

                swp['stimulus']['timeseries']['data'][...] = stimulus

            if response is not None:
                # if the data is shorter than the sweep, pad it with zeros
                missing_data = sweep_length - len(response)
                if missing_data > 0:
                    response = np.append(response, np.zeros(missing_data))


                swp['response']['timeseries']['data'][...] = response

    def get_spike_times(self, sweep_number):
        with h5py.File(self.file_name,'r') as f:
            sweep_name = "Sweep_%d" % sweep_number

            try:
                spikes = f["analysis"]["spike_times"][sweep_name]
            except KeyError:
                return []

            return spikes.value


    def set_spike_times(self, sweep_number, spike_times):
        with h5py.File(self.file_name,'r+') as f:
            # make sure expected directory structure is in place
            if "analysis" not in f.keys():
                f.create_group("analysis")

            analysis_dir = f["analysis"]
            if "spike_times" not in analysis_dir.keys():
                analysis_dir.create_group("spike_times")

            spike_dir = analysis_dir["spike_times"]

            # see if desired dataset already exists
            sweep_name = "Sweep_%d" % sweep_number
            if sweep_name in spike_dir.keys():
                # rewriting data -- delete old dataset
                del spike_dir[sweep_name]

            spike_dir.create_dataset(sweep_name, data=spike_times, dtype='f8')
            


# Copyright 2014 Allen Institute for Brain Science
# Licensed under the Allen Institute Terms of Use (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.alleninstitute.org/Media/policies/terms_of_use_content.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from allen_wrench.config.model.lob_parser import LobParser
import h5py
import logging


class OrcaLobParser(LobParser):
    def __init__(self):
        self.log = logging.getLogger(__name__)
    
    
    def read(self, file_path, *args, **kwargs):
        stim_curr = None
        sampling_rate = None
        
        # This way of accessing is based on KeithG's code that reads orca files
        sweep = kwargs.get('sweep', 0)        
        stim_name = "Experiment_%d" % sweep

        try:
            with h5py.File(file_path, 'r') as f:
                stim_epoch = f["epochs"][stim_name]
                
                # This extracts the injected current waveform (in amperes)
                #
                # NOTE: According to KeithG, in the Orca-0.3.x files, the "timeseries" group name
                timeseries_string = "timeseries"
                new_style = True                
                
                # TODO: deprecate
                if not "timeseries" in stim_epoch["stimulus"]:
                    timeseries_string = "sequence"
                    new_style = False
                
                
                stim_curr = stim_epoch["stimulus"][timeseries_string]["data"][0:]
                stim_curr *= 1e9 # convert to nA for NEURON
               
                # Sampling rate currently represented in seconds
                # NOTE: According to KeithG, in the Orca-0.3.x files, the sampling_rate
                # will be in units of Hz, not seconds
                # So - OLD: 5e-6; NEW: 200000 are equivalent
                sampling_rate = stim_epoch["stimulus"][timeseries_string]["sampling_rate"].value
                
                if new_style:
                    sampling_rate = 1e3 / sampling_rate
                else:
                    sampling_rate *= 1e3 # convert to ms for NEURON
                
                return stim_curr, sampling_rate
        except Exception:
            self.log.error("Couldn't read ORCA file: %s" % file_path)
            raise
    
        return stim_curr, sampling_rate

    
    def write(self, file_path, data, *args, **kwargs):
        sweep = kwargs.get('sweep', 0)        
        stim_name = "Experiment_%d" % sweep

        try:
            with h5py.File(file_path, 'a') as f:
                stim_epoch = f["epochs"][stim_name]
                
                # This extracts the injected current waveform (in amperes)
                #
                # NOTE: According to KeithG, in the Orca-0.3.x files, the "timeseries" group name
                timeseries_string = "timeseries"
                new_style = True                
                
                # TODO: deprecate
                if not "timeseries" in stim_epoch["response"]:
                    timeseries_string = "sequence"
                    new_style = False
                    
                stim_epoch["response"][timeseries_string]["data"][...] = data
        except Exception:
            self.log.error("Couldn't write ORCA file: %s" % file_path)
            raise

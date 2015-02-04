import h5py
import logging


class OrcaLobParser(object):
    log = logging.getLogger(__name__)
    
    def __init__(self):
        pass    
    

    def sweep_numbers(self, file_path):
        with h5py.File(file_path, 'r') as f:
            sweeps = [int(e.split('_')[1]) for e in f['epochs'].keys() if e.startswith('Experiment')]
        
                
        OrcaLobParser.log.info("sweeps: %s" % (','.join(str(s) for s in sweeps)))        

        return sweeps        

    
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
            OrcaLobParser.log.error("Couldn't read ORCA file: %s" % file_path)
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

                output_len = len(stim_epoch["response"][timeseries_string]["data"])
                data_len = len(data)
                if data_len <= output_len:
                    output_len = data_len
                else:
                    OrcaLobParser.log.warn(
                        'sweep %d output data is longer than available space, truncating: %d > %d' %
                        (sweep, data_len, output_len))
                    
                stim_epoch["response"][timeseries_string]["data"][...] = data[0:output_len]
        except Exception:
            OrcaLobParser.log.error("Couldn't write ORCA file: %s" % file_path)
            raise

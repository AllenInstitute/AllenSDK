import logging, os
from allensdk.config.manifest_builder import ManifestBuilder
from allensdk.config.manifest import Manifest
from allensdk.internal.api.queries.biophysical_module_reader import BiophysicalModuleReader

class SimulateManifestBuilder(object):
    _log = logging.getLogger(__name__)
    
    def __init__(self):
        pass
    
    
    def get_config(self):
        wrapper = { "manifest": self.path_info }
        wrapper.update(self.bps_cfg)
        wrapper.update(self.stimulus_conf)
        
        return wrapper
        
        
def lims_config(lims_json_path):    
    input_config = BiophysicalModuleReader()
    input_config.read_json(lims_json_path)
    
    return input_config
    
def write_bps_cfg(input_config):
    bps_cfg_template = ("[biophys]\n"
                        "log_config_path: logs.conf\n"
                        "model_file: manifest.json,{{FIT_JSON}}\n")
    with open('bps.cfg', 'wb') as f:
        f.write(bps_cfg_template.replace('{{FIT_JSON}}',
                                         input_config.fit_parameters_path(),
                                         1))
        

def lims_manifest(input_config):
    b = ManifestBuilder()
    b.add_path('BASEDIR', '.')
    
    b.add_path('WORKDIR',
               input_config.lims_working_directory())
        
    b.add_path('MORPHOLOGY',
               input_config.morphology_path(),
               typename='file')
    b.add_path('CODE_DIR', 'templates')
    b.add_path('MODFILE_DIR', 'modfiles')
    
    b.add_path('stimulus_path',
               input_config.stimulus_path(),
               typename='file')
    
    nwb_file_name, extension = \
        os.path.splitext(os.path.basename(input_config.stimulus_path()))                                     
    b.add_path('output_path',
               '%s_virtual_experiment%s' % (nwb_file_name, extension),
               typename='file',
               parent_key='WORKDIR')

    b.add_path('manifest',
               os.path.join(os.path.realpath(os.curdir), 'manifest.json'),
               typename='file')

    b.add_section('biophys',  {"biophys": [
                {"model_file": ["manifest.json",
                                input_config.fit_parameters_path()],
                 "model_type": input_config.model_type()}
                ]})

    
    b.add_section('stimulus_conf',
                  {"runs": [{"sweeps": input_config.sweep_numbers()
                             }]})

    b.add_section('hoc_conf',
                  {"neuron" : [{"hoc": [ "stdgui.hoc", "import3d.hoc", "cell.hoc" ]
                                }]})     
        
    m = Manifest(config=b.path_info)
    
    b.write_json_file('manifest.json')
    
    return m
    
if __name__ == '__main__':
    lims_json_path = 'from_lims/first_cluster_run/EPHYS_BIOPHYS_SIMULATE_QUEUE_397352299_input.js'
    config_from_lims = lims_config(lims_json_path)
    lims_manifest(config_from_lims)

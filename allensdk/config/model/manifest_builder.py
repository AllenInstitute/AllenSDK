from allensdk.config.model.manifest import Manifest
import allensdk.core.json_utilities as ju
from pkg_resources import resource_filename
import logging, os


class ManifestBuilder(object):
    def __init__(self):
        self._log = logging.getLogger(__name__)
        self.path_info = []
        self.bps_cfg = {}
        self.stimulus_conf = {}
        self.hoc_conf = {}
    
    
    def add_path(self, key, spec,
                 typename='dir',
                 parent_key=None,
                 format=None):
        entry = {
            'key': key,
            'type': typename,
            'spec': spec }
        
        if format != None:
            entry['format'] = format
        
        if parent_key != None:
            entry['parent_key'] = parent_key
            
        self.path_info.append(entry)
    
    
    def write_json_file(self, path):
        self.add_biophys_sim_config(path)
        
        with open(path, 'wb') as f:
            f.write(self.write_json_string())
    
    
    def get_config(self):
        wrapper = { "manifest": self.path_info }
        wrapper.update(self.bps_cfg)
        wrapper.update(self.stimulus_conf)
        wrapper.update(self.hoc_conf)
        
        return wrapper

    
    def add_biophys_sim_config(self, path):
        self.bps_cfg = {
            "biophys": [{
                "model_file": [ path ]
                }]
            }
    
    def write_json_string(self):
        config = self.get_config()
        return ju.write_string(config)
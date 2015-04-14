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

import logging


class LobParser(object):
    """ Reads and writes large objects (LOBs) that are inconvenient to store in
        a format supported by configuration parser formats. """
    log = logging.getLogger(__name__)
    
    def __init__(self):
        pass
    
    
    def read(self, file_path, *args, **kwargs):
        pass
    
    
    def write(self, file_path, data, *args, **kwargs):
        pass
    
    
    @classmethod
    def unpack_lobs(cls, manifest, configuration_data, sections):
        # Circular imports
        from allensdk.config.model.formats.hdf5_lob_parser import Hdf5LobParser
        
        for section in sections:
            if not section in configuration_data:
                LobParser.log.info("section %s not found, skipping." % (section))
                continue
            
            for entry in configuration_data[section]:
                if 'lob_source' in entry:
                    manifest_key = entry['lob_source']
                    path = manifest.get_path(manifest_key)
                    path_format = manifest.get_format(manifest_key)
                    
                    if path_format == 'hdf5':
                        reader = Hdf5LobParser()
                        try:
                            entry['data'] = reader.read(path)
                        except:
                            pass
                    else:
                        LobParser.log.warning("LOB format not recognized: %s" % 
                                              (path_format))

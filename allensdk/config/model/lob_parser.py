# Copyright 2014 Allen Institute for Brain Science
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

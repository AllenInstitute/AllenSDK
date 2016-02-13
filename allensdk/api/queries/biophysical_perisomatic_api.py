# Copyright 2015-2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


class BiophysicalPerisomaticApi(object):
    
    def __init__(self, base_uri=None):
        raise(Exception("BiophysicalPerisomaticApi has been renamed BiophysicalApi"))
    
    
    def build_rma(self, neuronal_model_id, fmt='json'):
        raise(Exception("BiophysicalPerisomaticApi has been renamed BiophysicalApi"))
    
    
    def read_json(self, json_parsed_data):
        raise(Exception("BiophysicalPerisomaticApi has been renamed BiophysicalApi"))
    
    
    def is_well_known_file_type(self, wkf, name):
        raise(Exception("BiophysicalPerisomaticApi has been renamed BiophysicalApi"))
    
    
    def get_well_known_file_ids(self, neuronal_model_id):
        raise(Exception("BiophysicalPerisomaticApi has been renamed BiophysicalApi"))
    
    
    def create_manifest(self,
                        fit_path='',
                        model_type='',
                        stimulus_filename='',
                        swc_morphology_path='',
                        marker_path='',
                        sweeps=[]):
        raise(Exception("BiophysicalPerisomaticApi has been renamed BiophysicalApi"))
    
    
    def cache_data(self,
                   neuronal_model_id,
                   working_directory=None):
        raise(Exception("BiophysicalPerisomaticApi has been renamed BiophysicalApi"))

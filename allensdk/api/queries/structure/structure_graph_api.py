# Copyright 2015 Allen Institute for Brain Science
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

from allensdk.api.api import Api

class StructureGraphApi(Api):
    def __init__(self, base_uri=None):
        super(StructureGraphApi, self).__init__(base_uri)
    
    
    def build_structure_graph_query(self,
                                    structure_graph_id,
                                    fmt='json'):
        '''Build the URL that will fetch meta data for the specified structure graph.
        
        Parameters
        ----------
        structure_graph_id : integer
            what to retrieve
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.structure_graph_endpoint,
                       '/',
                       str(structure_graph_id),
                       '.',
                       fmt])
        
        return url
    
    
    def read_graph(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def get_structure_graph_by_id(self, structure_graph_id):
        '''Retrieve the structure graph data.'''
        graph_data = self.do_query(self.build_structure_graph_query,
                                   self.read_graph,
                                   structure_graph_id)
        
        return graph_data

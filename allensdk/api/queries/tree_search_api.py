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

class TreeSearchApi(Api):
    def __init__(self, base_uri=None):
        super(TreeSearchApi, self).__init__(base_uri)
    
    
    def build_query(self, kind, db_id, fmt='json', ancestors=None, descendants=None):
        '''Build the URL that will fetch meta data for the specified structure.
        
        Parameters
        ----------
        kind : string
            'Structure' or 'Specimen'
        db_id : integer
            The id of the structure or specimen to search.
        fmt : string, optional
            json (default) or xml
        ancestors : boolean, optional
            whether to include ancestors in the response (defaults to False)
        descendants : boolean, optional
            whether to include descendants in the response (defaults to False)
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        params = []
        url_params = ''
        
        if ancestors == True:
            params.append('ancestors=true')
        elif ancestors == False:
            params.append('ancestors=false')
        
        if descendants == True:
            params.append('descendants=true')
        elif descendants == False:
            params.append('descendants=false')
        
        if len(params) > 0:
            url_params = '?' + '&'.join(params)
        else:
            url_params = ''
        
        url = ''.join([self.tree_search_endpoint,
                       '/',
                       kind,
                       '/',
                       str(db_id),
                       '.',
                       fmt,
                       url_params])
        
        return url
    
    
    def read_result(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def get_structure_tree_by_id(self, structure_id, ancestors=None, descendants=None):
        '''Retrieve the structure tree data.'''
        tree_data = self.do_query(self.build_query,
                                  self.read_result,
                                  'Structure',
                                  structure_id,
                                  ancestors=ancestors,
                                  descendants=descendants)
        
        return tree_data
    
    
    def get_specimen_tree_by_id(self, specimen_id, ancestors=None, descendants=None):
        '''Retrieve the specimen tree data.'''
        tree_data = self.do_rma_query(self.build_query,
                                       self.read_result,
                                       'Specimen',
                                       specimen_id,
                                       ancestors=ancestors,
                                       descendants=descendants)
        
        return tree_data


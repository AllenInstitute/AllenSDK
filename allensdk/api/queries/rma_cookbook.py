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

from allensdk.api.queries.rma_api import RmaApi

class RmaCookbook(RmaApi):
    '''
    See: `Atlas Drawings and Ontologies
    <http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies>`_
    '''
    
    def __init__(self, base_uri=None, query_manifest=None):
        super(RmaCookbook, self).__init__(base_uri)
        self.cookbook = query_manifest
    
    
    def cookbook_query(self, cookbook_name, entry_name, **kwargs):
        cb = self.cookbook[cookbook_name]
        templates = [e for e in cb if e['name'] == entry_name]
        
        if len(templates) > 0:
            template = templates[0]
        else:
            raise Exception('Entry %s not found.' % (entry_name))
        
        query_args = {'model': template['model']}
        
        if 'criteria' in template:
            criteria_string = template['criteria']
            
            if 'criteria_params' in template:
                criteria_string = criteria_string % tuple(kwargs.get(key) for key in template['criteria_params'])
            
            query_args['criteria'] = criteria_string
        
        if 'include' in template:
            include_string = template['include']
            
            if 'include_params' in template:
                include_string = include_string % tuple(kwargs.get(key) for key in template['include_params'])
            
            query_args['include'] = include_string
        
        if 'only' in template:
            query_args['only'] = template['only']
        
        if 'except' in template:
            query_args['except'] = template['except']
        
        if 'num_rows' in template:
            query_args['num_rows'] = template['num_rows']
        
        if 'count' in template:
            query_args['count'] = template['count']
        
        url = self.model_query(**query_args)
        
        return url


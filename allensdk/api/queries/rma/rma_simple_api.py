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

from allensdk.api.queries.rma.rma_api import RmaApi

class RmaSimpleApi(RmaApi):
    def __init__(self, base_uri=None):
        super(RmaSimpleApi, self).__init__(base_uri)
    
    
    def read_data(self, parsed_json):
        return parsed_json['msg']
    
    
    def model_query(self, *args, **kwargs):
        '''
        Parameters
        ----------
        model : string
        filters :
        criteria :
        include :
        '''
        return self.do_query(
            lambda *a, **k: self.build_query_url(self.model_stage(*a, **k)),
            self.read_data,
            *args,
            **kwargs)
    
    
    def service_query(self, *args, **kwargs):
        return self.do_query(
            lambda *a, **k: self.build_query_url(self.service_stage(*a, **k)),
            self.read_data,
            *args,
            **kwargs)

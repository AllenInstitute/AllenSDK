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

from ..api import Api


class TreeSearchApi(Api):
    '''

    See `Searching a Specimen or Structure Tree <http://help.brain-map.org/display/api/Image-to-Image+Synchronization>`_
    for additional documentation.
    '''

    def __init__(self, base_uri=None):
        super(TreeSearchApi, self).__init__(base_uri)

    def get_tree(self,
                 kind,
                 db_id,
                 ancestors=None,
                 descendants=None):
        '''Fetch meta data for the specified structure or specimen.

        Parameters
        ----------
        kind : string
            'Structure' or 'Specimen'
        db_id : integer
            The id of the structure or specimen to search.
        ancestors : boolean, optional
            whether to include ancestors in the response (defaults to False)
        descendants : boolean, optional
            whether to include descendants in the response (defaults to False)

        Returns
        -------
        dict
            parsed json response data
        '''
        params = []
        url_params = ''

        if ancestors is True:
            params.append('ancestors=true')
        elif ancestors is False:
            params.append('ancestors=false')

        if descendants is True:
            params.append('descendants=true')
        elif descendants is False:
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
                       '.json',
                       url_params])

        return self.json_msg_query(url)

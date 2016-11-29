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


class SvgApi(Api):

    def __init__(self, base_uri=None):
        super(SvgApi, self).__init__(base_uri)

    def build_query(self, section_image_id, groups=None, download=False):
        '''Build the URL that will fetch meta data for the specified structure.

        Parameters
        ----------
        section_image_id : integer
            Key of the object to be retrieved.
        groups : array of integers
            Keys of the group labels to filter the svg types that are returned.

        Returns
        -------
        url : string
            The constructed URL
        '''
        if download is True:
            endpoint = self.svg_download_endpoint
        else:
            endpoint = self.svg_endpoint

        if groups is None:
            groups = []

        if groups and len(groups) > 0:
            url_params = '?groups=' + ','.join([str(g) for g in groups])
        else:
            url_params = ''

        url = ''.join([endpoint,
                       '/',
                       str(section_image_id),
                       url_params])

        return url

    def download_svg(self,
                     section_image_id,
                     groups=None,
                     file_path=None):
        '''Download the svg file'''
        if file_path is None:
            file_path = '%d.svg' % (section_image_id)

        svg_url = self.build_query(section_image_id, groups, download=True)
        self.retrieve_file_over_http(svg_url, file_path)

    def get_svg(self,
                section_image_id,
                groups=None):
        '''Get the svg document.'''
        svg_url = self.build_query(section_image_id, groups)

        return self.retrieve_xml_over_http(svg_url)

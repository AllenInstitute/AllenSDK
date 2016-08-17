# Copyright 2014 Allen Institute for Brain Science
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

import logging
from pprint import pprint, pformat
from allensdk.config.model.description import Description
from allensdk.config.model.description_parser import DescriptionParser


class PycfgDescriptionParser(DescriptionParser):
    log = logging.getLogger(__name__)

    def __init__(self):
        super(PycfgDescriptionParser, self).__init__()

    def read(self, pycfg_file_path, description=None, section=None, **kwargs):
        '''Read a serialized description from a Python (.pycfg) file.

        Parameters
        ----------
        filename : string
            Name of the .pycfg file.

        Returns
        -------
        Description
            Configuration object.
        '''
        header = kwargs.get('prefix', '')

        with open(pycfg_file_path, 'r') as f:
            return self.read_string(f.read(), description, section, header=header)

    def read_string(self, python_string, description=None, section=None, **kwargs):
        '''Read a serialized description from a Python (.pycfg) string.

        Parameters
        ----------
        python_string : string
            Python string with a serialized description.

        Returns
        -------
        Description
            Configuration object.
        '''

        if description is None:
            description = Description()

        header = kwargs.get('header', '')

        python_string = "%s\n\nallensdk_description = %s" % (
            header, python_string)

        ns = {}
        code = compile(python_string, 'string', 'exec')
        exec(code, ns)
        data = ns['allensdk_description']
        description.unpack(data, section)

        return description

    def write(self, filename, description):
        '''Write the description to a Python (.pycfg) file.

        Parameters
        ----------
        filename : string
            Name of the file to write.
        '''
        try:
            with open(filename, 'w') as f:
                pprint(description.data, f, indent=2)

        except Exception:
            self.log.warn(
                "Couldn't write allensdk python description: %s" % filename)
            raise

        return

    def write_string(self, description):
        '''Write the description to a pretty-printed Python string.

        Parameters
        ----------
        description : Description
            Configuration object to write.
        '''
        pycfg_string = pformat(description.data, indent=2)

        return pycfg_string

# Copyright 2014-2016 Allen Institute for Brain Science
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
from allensdk.config.model.description import Description


class DescriptionParser(object):
    log = logging.getLogger(__name__)

    def __init__(self):
        pass

    def read(self, file_path, description=None, section=None, **kwargs):
        '''Parse data needed for a simulation.

        Parameters
        ----------
        description : dict
            Configuration from parsing previous files.
        section : string, optional
            What configuration section to read it into if the file does not specify.
        '''
        if description is None:
            description = Description()

        self.reader = self.parser_for_extension(file_path)
        self.reader.read(file_path, description, section, **kwargs)

        return description

    def read_string(self, data_string, description=None, section=None, header=None):
        '''Parse data needed for a simulation from a string.'''
        raise Exception("Not implemented, use a sub class")

    def write(self, filename, description):
        """Save the configuration.

        Parameters
        ----------
        filename : string
            Name of the file to write.
        """
        writer = self.parser_for_extension(filename)

        writer.write(filename, description)

    def parser_for_extension(self, filename):
        '''Choose a subclass that can read the format.

        Parameters
        ----------
        filename : string
           For the extension.

        Returns
        -------
        DescriptionParser
            Appropriate subclass.
        '''
        # Circular imports
        from allensdk.config.model.formats.json_description_parser import JsonDescriptionParser
        from allensdk.config.model.formats.pycfg_description_parser import PycfgDescriptionParser

        parser = None

        if filename.endswith('.json'):
            parser = JsonDescriptionParser()
        elif filename.endswith('.pycfg'):
            parser = PycfgDescriptionParser()
        else:
            raise Exception('could not determine file format')

        return parser

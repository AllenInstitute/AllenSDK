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
from json import dump, dumps
from allensdk.config.model.description_parser import DescriptionParser
from allensdk.config.model.description import Description
from allensdk.core.json_utilities import JsonComments


class JsonDescriptionParser(DescriptionParser):
    log = logging.getLogger(__name__)

    def __init__(self):
        super(JsonDescriptionParser, self).__init__()

    def read(self, file_path, description=None, section=None, **kwargs):
        '''Parse a complete or partial configuration.

        Parameters
        ----------
        json_string : string
            Input to parse.
        description : Description, optional
            Where to put the parsed configuration.  If None a new one is created.
        section : string, optional
            Where to put the parsed configuration within the description.

        Returns
        -------
        Description
            The input description with parsed configuration added.

        Section is only specified for "bare" objects that are to be added to a section array.
        '''
        if description is None:
            description = Description()

        data = JsonComments.read_file(file_path)
        description.unpack(data, section)

        return description

    def read_string(self, json_string, description=None, section=None, **kwargs):
        '''Parse a complete or partial configuration.

        Parameters
        ----------
        json_string : string
            Input to parse.
        description : Description, optional
            Where to put the parsed configuration.  If None a new one is created.
        section : string, optional
            Where to put the parsed configuration within the description.

        Returns
        -------
        Description
            The input description with parsed configuration added.

        Section is only specified for "bare" objects that are to be added to a section array.
        '''
        if description is None:
            description = Description()

        data = JsonComments.read_string(json_string)

        description.unpack(data, section)

        return description

    def write(self, filename, description):
        '''Write the description to a JSON file.

        Parameters
        ----------
        description : Description
            Object to write.
        '''
        try:
            with open(filename, 'w') as f:
                dump(description.data, f, indent=2)

        except Exception:
            self.log.warn(
                "Couldn't write allensdk json description: %s" % filename)
            raise

        return

    def write_string(self, description):
        '''Write the description to a JSON string.

        Parameters
        ----------
        description : Description
            Object to write.

        Returns
        -------
        string
           JSON serialization of the input.
        '''
        try:
            json_string = dumps(description.data,
                                indent=2)
            return json_string
        except Exception:
            self.log.warn("Couldn't write allensdk json description: %s")
            raise

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

from allensdk.config.manifest import Manifest


class Description(object):
    _log = logging.getLogger(__name__)

    def __init__(self):
        self.data = {}
        self.reserved_data = []
        self.manifest = Manifest()

    def update_data(self, data, section=None):
        '''Merge configuration data possibly from multiple files.

        Parameters
        ----------
        data : dict
            Configuration structure to add.
        section : string, optional
            What configuration section to read it into if the file does not specify.
        '''
        if section is None:
            for (section, entries) in data.items():
                if section not in self.data:
                    self.data[section] = entries
                else:
                    self.data[section].extend(entries)
        else:
            if section not in self.data:
                self.data[section] = []

            self.data[section].append(data)

    def is_empty(self):
        '''Check if anything is in the object.

        Returns
        -------
        boolean
            true if self.data is missing or empty
        '''
        if self.data:
            return False

        return True

    def unpack(self, data, section=None):
        '''Read the manifest and other stand-alone configuration structure,
        or insert a configuration object into a section of an existing configuration.

        Parameters
        ----------
        data : dict
            A configuration object including top level sections,
            or an configuration object to be placed within a section.
        section : string, optional.
            If this is present, place data within an existing section array.
        '''
        if section is None:
            self.unpack_manifest(data)
            self.update_data(data)
        else:
            self.update_data(data, section)

    def unpack_manifest(self, data):
        '''Pull the manifest configuration section into a separate place.

        Parameters
        ----------
        data : dict
            A configuration structure that still has a manifest section.
        '''
        data_manifest = data.pop("manifest", {})
        reserved_data = {"manifest": data_manifest}
        self.reserved_data.append(reserved_data)
        self.manifest.load_config(data_manifest)

    def fix_unary_sections(self, section_names=None):
        ''' Wrap section contents that don't have the proper
        array surrounding them in an array.

        Parameters
        ----------
        section_names : list of strings, optional
            Keys of sections that might not be in array form.
        '''
        if section_names is None:
            section_names = []

        for section in section_names:
            if section in self.data:
                if type(self.data[section]) is dict:
                    self.data[section] = [self.data[section]]
                    Description._log.warn(
                        "wrapped description section %s in an array." % (section))

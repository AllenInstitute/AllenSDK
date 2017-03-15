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

import allensdk.core.json_utilities as ju
import logging
from allensdk.config.manifest import Manifest
import pandas as pd
import six


class ManifestBuilder(object):
    df_columns = ['key', 'parent_key', 'spec', 'type', 'format']

    def __init__(self):
        self._log = logging.getLogger(__name__)
        self.path_info = []
        self.sections = {}

    def set_version(self, value):
        self.path_info.append({'type': Manifest.VERSION, 'value': value})

    def add_path(self, key, spec,
                 typename='dir',
                 parent_key=None,
                 format=None):
        entry = {
            'key': key,
            'type': typename,
            'spec': spec}

        if format is not None:
            entry['format'] = format

        if parent_key is not None:
            entry['parent_key'] = parent_key

        self.path_info.append(entry)

    def add_section(self, name, contents):
        self.sections[name] = contents

    def write_json_file(self, path, overwrite=False):
        mode = 'wb'

        if overwrite is True:
            mode = 'wb+'

        json_string = self.write_json_string()

        with open(path, mode) as f:
            try:
                f.write(json_string)   # Python 2.7
            except TypeError:
                f.write(bytes(json_string, 'utf-8'))  # Python 3

    def get_config(self):
        wrapper = {"manifest": self.path_info}
        for section in self.sections.values():
            wrapper.update(section)

        return wrapper

    def get_manifest(self):
        return Manifest(self.path_info)

    def write_json_string(self):
        config = self.get_config()
        return ju.write_string(config)

    def as_dataframe(self):
        return pd.DataFrame(self.path_info,
                            columns=ManifestBuilder.df_columns)

    def from_dataframe(self, df):
        self.path_info = {}

        for _, k, p, s, t, f in six.iteritems(df.loc[:, ManifestBuilder.df_columns]):
            self.add_path(k, s, typename=t, parent=p, format=f)

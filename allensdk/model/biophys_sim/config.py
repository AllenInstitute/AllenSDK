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

import re
import logging
from pkg_resources import resource_filename  # @UnresolvedImport
from allensdk.config.app.application_config import ApplicationConfig
from allensdk.config.model.description_parser import DescriptionParser
from allensdk.config.model.description import Description


class Config(ApplicationConfig):
    _log = logging.getLogger(__name__)

    _DEFAULT_LOG_CONFIG = resource_filename(__name__, 'logging.conf')

    #: A structure that defines the available configuration parameters.
    #: The default value and help strings may be seen by viewing the source.
    _DEFAULTS = {
        'workdir': {'default': 'workdir',
                    'help': 'writable directory where intermediate and output files are written.'},
        'data_dir': {'default': '',
                     'help': 'writable directory where intermediate and output files are written.'},
        'model_file': {'default': 'config.json',
                       'help': 'file where the model parameters are set.'},
        'main': {'default': 'simulation#run',
                 'help': 'module#function that runs the actual simulation'}
    }

    def __init__(self):
        super(Config, self).__init__(Config._DEFAULTS,
                                     name='biophys',
                                     halp='tools for biophysically detailed modeling at the Allen Institute.',
                                     default_log_config=Config._DEFAULT_LOG_CONFIG)

    def load(self, config_path,
             disable_existing_logs=False):
        '''Parse the application configuration then immediately load
        the model configuration files.

        Parameters
        ----------
        disable_existing_logs : boolean, optional
            If false (default) leave existing logs after configuration.
        '''
        super(Config, self).load([config_path], disable_existing_logs)
        description = self.read_model_description()

        return description

    def read_model_description(self):
        '''parse the model_file field of the application configuration
        and read the files.

        The model_file field of the application configuration is
        first split at commas, since it may list more than one file.

        The files may be uris of the form :samp:`file:filename?section=name`,
        in which case a bare configuration object is read from filename
        into the configuration section with key 'name'.

        A simple filename without a section option
        is treated as a standard multi-section configuration file.

        Returns
        -------
        description : Description
            Configuration object.
        '''
        reader = DescriptionParser()
        description = Description()

        Config._log.info("model file: %s" % self.model_file)

        # TODO: make space aware w/ regex
        for model_file in self.model_file.split(','):
            if not model_file.startswith("file:"):
                model_file = 'file:' + model_file

            file_regex = re.compile(r"^file:([^?]*)(\?(.*)?)?")
            m = file_regex.match(model_file)
            model_file = m.group(1)
            file_url_params = {}
            if m.group(3):
                file_url_params.update(((x[0], x[1])
                                        for x in (y.split('=')
                                                  for y in m.group(3).split('&'))))
            if 'section' in file_url_params:
                section = file_url_params['section']
            else:
                section = None
            Config._log.info("reading model file %s" % (model_file))
            reader.read(model_file, description, section)

        return description

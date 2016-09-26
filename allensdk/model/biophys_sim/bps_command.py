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

import os
import subprocess as sp
import logging
from .config import Config


def choose_bps_command(command='bps_simple', conf_file=None):
    log = logging.getLogger('allensdk.model.biophys_sim.bps_command')

    log.info("bps command: %s" % (command))

    if conf_file:
        conf_file = os.path.abspath(conf_file)

    if command == 'help':
        print(Config().argparser.parse_args(['--help']))
    elif command == 'nrnivmodl':
        sp.call(['nrnivmodl', 'modfiles'])
    elif command == 'run_simple':
        app_config = Config()
        description = app_config.load(conf_file)
        sys.path.insert(1, description.manifest.get_path('CODE_DIR'))
        (module_name, function_name) = description.data[
            'runs'][0]['main'].split('#')
        run_module(description, module_name, function_name)
    else:
        raise Exception("unknown command %s" % (command))


def run_module(description, module_name, function_name):
    m = __import__(module_name, fromlist=[function_name])
    func = getattr(m, function_name)

    func(description)


# this module is designed to be called from the bps script,
# which may use nrniv which does not pass in command line arguments
# So the configuration file path must be set in an environment variable.
if __name__ == '__main__':
    import sys
    conf_file = None
    argv = sys.argv

    if len(argv) > 1:
        if argv[0] == 'nrniv':
            command = 'run_simple'
        else:
            command = argv[1]
    else:
        command = 'run_simple'

    if len(argv) > 2 and (argv[-1].endswith('.conf') or
                          argv[-1].endswith('.json')):
        conf_file = argv[-1]
    else:
        try:
            conf_file = os.environ['CONF_FILE']
        except:
            pass

    choose_bps_command(command, conf_file)

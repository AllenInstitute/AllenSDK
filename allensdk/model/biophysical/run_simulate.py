# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

from . import runner as single_cell
import logging
import os
import sys
import traceback
import subprocess
import logging.config as lc
from ..biophys_sim.config import Config
from pkg_resources import resource_filename  # @UnresolvedImport


class RunSimulate(object):
    _log = logging.getLogger('allensdk.model.biophysical.run_simulate')

    def __init__(self,
                 input_json,
                 output_json):
        self.input_json = input_json
        self.output_json = output_json
        self.app_config = None
        self.manifest = None

    def load_manifest(self):
        self.app_config = Config().load(self.input_json)
        self.manifest = self.app_config.manifest
        fix_sections = ['passive', 'axon_morph,', 'conditions', 'fitting']
        self.app_config.fix_unary_sections(fix_sections)

    def nrnivmodl(self):
        RunSimulate._log.debug("nrnivmodl")

        subprocess.call(['nrnivmodl', './modfiles'])

    def simulate(self):
        from allensdk.internal.api.queries.biophysical_module_reader \
            import BiophysicalModuleReader

        self.load_manifest()

        try:
            stimulus_path = self.manifest.get_path('stimulus_path')
            RunSimulate._log.info("stimulus path: %s" % (stimulus_path))
        except:
            raise Exception(
                'Could not read input stimulus path from input config.')

        try:
            out_path = self.manifest.get_path('output_path')
            RunSimulate._log.info("result NWB file: %s" % (out_path))
        except:
            raise Exception('Could not read output path from input config.')

        try:
            morphology_path = self.manifest.get_path('MORPHOLOGY')
            RunSimulate._log.info("morphology path: %s" % (morphology_path))
        except:
            raise Exception(
                'Could not read morphology path from input config.')

        single_cell.run(self.app_config)

        lims_upload_config = BiophysicalModuleReader()
        lims_upload_config.read_json(
            self.manifest.get_path('neuronal_model_run_data'))
        lims_upload_config.update_well_known_file(out_path)
        lims_upload_config.set_workflow_state('passed')
        lims_upload_config.write_file(self.output_json)


def main(command, lims_strategy_json, lims_response_json):
    ''' Entry point for module.
        :param command: select behavior, nrnivmodl or simulate
        :type command: string
        :param lims_strategy_json: path to json file output from lims.
        :type lims_strategy_json: string
        :param lims_response_json: path to json file returned to lims.
        :type lims_response_json: string
    '''
    rs = RunSimulate(lims_strategy_json,
                     lims_response_json)

    RunSimulate._log.debug("command: %s" % (command))
    RunSimulate._log.debug("lims strategy json: %s" % (lims_strategy_json))
    RunSimulate._log.debug("lims upload json: %s" % (lims_response_json))

    log_config = resource_filename('allensdk.model.biophysical.run_simulate',
                                   'logging.conf')
    lc.fileConfig(log_config)
    os.environ['LOG_CFG'] = log_config

    if 'nrnivmodl' == command:
        rs.nrnivmodl()
    else:
        rs.simulate()


if __name__ == '__main__':
    command, input_json, output_json = sys.argv[-3:]

    try:
        main(command, input_json, output_json)
        RunSimulate._log.debug("success")
    except Exception as e:
        RunSimulate._log.error(traceback.format_exc())
        exit(1)

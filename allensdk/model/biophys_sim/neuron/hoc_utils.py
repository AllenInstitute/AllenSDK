# Copyright 2015-2016 Allen Institute for Brain Science
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


class HocUtils(object):
    '''A helper class for containing references to NEUORN.

    Attributes
    ----------
    h : object
        The NEURON hoc object.
    nrn : object
        The NEURON python object.
    neuron : module
        The NEURON module.
    '''
    _log = logging.getLogger(__name__)
    h = None
    nrn = None
    neuron = None

    def __init__(self, description):
        import neuron
        import nrn

        self.h = neuron.h
        HocUtils.neuron = neuron
        HocUtils.nrn = nrn
        HocUtils.h = self.h

        self.description = description
        self.manifest = description.manifest

        self.hoc_files = description.data['neuron'][0]['hoc']

        self.initialize_hoc()

    def initialize_hoc(self):
        '''Basic setup for NEURON.'''
        h = self.h
        params = self.description.data['conditions'][0]

        for hoc_file in self.hoc_files:
            HocUtils._log.info("loading hoc file %s" % (hoc_file))
            HocUtils.h.load_file(str(hoc_file))

        h('starttime = startsw()')

        if 'celsius' in params:
            h.celsius = params['celsius']

        if 'v_init' in params:
            h.v_init = params['v_init']

        if 'dt' in params:
            h.dt = params['dt']
            h.steps_per_ms = 1.0 / h.dt

        if 'tstop' in params:
            h.tstop = params['tstop']
            h.runStopAt = h.tstop

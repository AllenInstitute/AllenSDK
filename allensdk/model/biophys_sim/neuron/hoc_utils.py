from pkg_resources import resource_filename
import logging


class HocUtils(object):
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
        ''' :parameter params: a dict of key-values
        '''
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
            h.steps_per_ms = 1.0/h.dt
            
        if 'tstop' in params:
            h.tstop = params['tstop']
            h.runStopAt = h.tstop
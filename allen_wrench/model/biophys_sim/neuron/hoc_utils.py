from pkg_resources import resource_filename
import logging


class HocUtils(object):
    log = logging.getLogger(__name__)
    h = None
    nrn = None
    neuron = None
    
    def __init__(self, manifest):
        self.manifest = manifest
        
        self.hoc_3d_files = ['import3d_sec.hoc',
                             'read_swc.hoc',
                             'read_nlcda.hoc',
                             'read_nlcda3.hoc',
                             'read_nts.hoc',
                             'read_morphml.hoc',
                             'import3d_gui.hoc']
        
        self.hoc_files = ['log.hoc',
                          'parlib.hoc',
                          'import3d.hoc',
                          'distSynsUniform_caa_as.hoc']
        
        self.hoc_lib_files = ['progress.hoc',
                              'save_t_series.hoc',
                              'spikefile.hoc',
                              'mkstim-SEClamp.hoc']
            
    
    def load_hoc_files(self, package, files):
        for hoc_file in files:
            HocUtils.h.load_file(resource_filename(package, hoc_file))
    
    
    def initialize_hoc(self, params):
        ''' :parameter params: a dict of key-values
        '''
        import neuron
        import nrn
        
        h = neuron.h
        HocUtils.neuron = neuron
        HocUtils.nrn = nrn
        HocUtils.h = h
        h.load_file('stdgui.hoc')

        self.load_hoc_files('allen_wrench.model.biophys_sim.neuron_hoc.import3d', self.hoc_3d_files)
        self.load_hoc_files('allen_wrench.model.biophys_sim.neuron_hoc.common', self.hoc_files)
        self.load_hoc_files('allen_wrench.model.biophys_sim.neuron_hoc', self.hoc_lib_files)
        
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
        
        return h

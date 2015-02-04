from allen_wrench.model.single_cell_biophysical.orca_lob_parser \
    import OrcaLobParser
import logging


class IclampStimulus(object):
    log = logging.getLogger(__name__)
    
    def __init__(self, h):
        super(IclampStimulus, self).__init__()
        
        self.h = h
        self.stim = None
        self.stim_curr = None
        self.sampling_rate = None

        self.stim_vec_list = []
    

    def setup_instance(self, *args, **kwargs):
        self.setup_instance_orca(*args, **kwargs)
                        
                
    def setup_instance_orca(self,
                            stimulus_path,
                            sweep=0):
          
        # work around pandas 0.14 issue w/ creating int columns             
        # e.g. target_cell_gid = int(target_cell_gid)
        
        self.stim = self.h.IClamp(self.h.soma[0](0.5))  # TODO: does soma have to be parametrized?
        self.stim.amp = 0
        self.stim.delay = 0
        self.stim.dur = 1e12 # just set to be really big; doesn't need to match the waveform

        self.read_stimulus(stimulus_path, sweep=sweep)
        self.h.dt = self.sampling_rate
        stim_vec = self.h.Vector(self.stim_curr)
        stim_vec.play(self.stim._ref_amp, self.sampling_rate)
        
        self.h.tstop = len(self.stim_curr) * self.sampling_rate
        # self.h.tstop = params['stimulus']['tstop']  # TODO: make sure this is done at the top level.
        self.stim_vec_list.append(stim_vec)
                

    def read_stimulus(self, stimulus_path, sweep=0):
        orca_parser = OrcaLobParser()
        (self.stim_curr, self.sampling_rate) = orca_parser.read(stimulus_path, sweep=sweep)
        
from allen_wrench.config.model.lob_parser import LobParser
import numpy

class FTrainDatLobParser(LobParser):
    def __init__(self):
        pass
    
    def read(self, file_path):
        spike_trains = []
        
        with open(file_path, 'r') as f:
            for line in f:
                # first entry is redundant n_spikes
                # convert the rest to an array of floats
                spike_t = numpy.array([float(x) for x in line.split()][1:])
                spike_trains.append(spike_t) # 0 is redunand n_spikes
                
        return spike_trains

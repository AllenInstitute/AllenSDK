from allen_wrench.model.biophys_sim.neuron.hoc_utils import HocUtils

class ParallelContext(object):
    def __init__(self):
        self.hoc_object = HocUtils.h
        self.hoc_parallel_context = self.hoc_object.ParallelContext() # from parlib.hoc
        self.rank = int(self.hoc_parallel_context.id())
        self.nhost = int(self.hoc_parallel_context.nhost())
        
        self.nclist = []
        
        
    def barrier(self):
        self.hoc_parallel_context.barrier()
        
        
    def timeout(self, t):
        self.hoc_parallel_context.timeout(t)
        
        
    def gid2cell(self, gid):
        return self.hoc_parallel_context.gid2cell(gid)
        
        
    def gid_exists(self, cell_gid):
        return self.hoc_parallel_context.gid_exists(cell_gid)
    
    
    def local_gids(self, gids):
        return [gid for gid in gids if self.gid_exists(gid)]
    
    
    def gid_connect(self, source_gid, target_gid):
        return self.hoc_parallel_context.gid_connect(source_gid, target_gid)
    
    
    def nclist_append(self, nc):
        self.nclist.append(nc)

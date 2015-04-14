class Progress(object):
    def __init__(self, init_k=0, skip=1, total=100.0, log=None, message="Working"):
        self.k = init_k
        self.skip = skip
        self.total = total
        self.log = log
        self.message = message
        
    def tick(self, k=None):
        if k == None:
            k = self.k
            
        if k % self.skip == 0:
            percent = 100.0 * k / self.total
            self.log.info("%s; progress: %.5f percent." % 
                          (self.message, percent))
        
        self.k = k + 1



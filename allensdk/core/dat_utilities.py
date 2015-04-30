import numpy

class DatUtilities(object):
    
    @classmethod
    def save_voltage(cls, output_path, v, t):
        '''Save a single voltage output result into a simple text format.
        
        The output file is one t v pair per line.
        
        Parameters
        ----------
        output_path : string
            file name for output
        v : numpy array
            voltage
        t : numpy array
            time
        '''
        data = numpy.transpose(numpy.vstack((t, v)))
        with open (output_path, "w") as f:
            numpy.savetxt(f, data)
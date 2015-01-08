class EphysDataSet( object ):
    def __init__(self, file_name):
        self.file_name = file_name
        
    def get_test_pulse(sweep_number):
        raise Exception("get_test_pulse not implemented")

    def get_experiment(sweep_number):
        raise Exception("get_experiment not implemented")
    
    def get_full_sweep(sweep_number):
        raise Exception("get_full_sweep not implemented")

    def set_test_pulse(sweep_number, stimulus, response):
        raise Exception("get_test_pulse not implemented")

    def set_experiment(sweep_number, stimulus, response):
        raise Exception("get_experiment not implemented")
    
    def set_full_sweep(sweep_number, stimulus, response):
        raise Exception("get_full_sweep not implemented")
    

        
        

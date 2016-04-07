import logging

def config_logging(loggers=['allensdk'],
                   level=logging.DEBUG,
                   log_file = 'biophysical.log'):    
    if config_logging.logging_configured == False:
        fh = logging.FileHandler(log_file)            
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        for logger in loggers:
            lims2_module_log = logging.getLogger(logger)
            lims2_module_log.setLevel(logging.DEBUG)
            lims2_module_log.addHandler(fh)
            lims2_module_log.propagate = False

        root_log = logging.getLogger()
        root_log.setLevel(logging.DEBUG)
        console_handler = None
        
        if len(root_log.handlers) > 0:
            console_handler = root_log.handlers[0]

        root_log.addHandler(fh)
        
        if console_handler != None:    
            root_log.removeHandler(console_handler)
            
        root_log.propagate = False
        root_log.disabled = True
                
        config_logging.logging_configured = True
        
config_logging.logging_configured = False

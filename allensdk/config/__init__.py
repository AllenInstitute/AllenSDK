import logging
import sys


def enable_console_log(level=None):
    '''configure allensdk logging to output to the console.

        Parameters
        ----------
        level : int
            logging level 0-50 (logging.INFO, logging.DEBUG, etc.) 

        See: `Logging Cookbook <https://docs.python.org/2/howto/logging-cookbook.html>`_.
    '''
    sdk_logger = logging.getLogger('allensdk')

    if level is None:
        sdk_logger.setLevel(logging.INFO)
    else:
        sdk_logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    sdk_logger.addHandler(console_handler)
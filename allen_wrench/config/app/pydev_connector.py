from urlparse import urlparse, parse_qs
import logging

class PydevConnector(object):
    _log = logging.getLogger(__name__)
    
    @classmethod
    def connect(cls, url=None):
        ''' Convenience method for connecting to the Eclipse/pydev remote debugger.
            see: http://pydev.org/manual_adv_remote_debugger.html
        '''
        if url == None:
            url = 'pydev://localhost:5678'
        
        try:
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            port = parsed_url.port
            
            suspend = False
            queries = parse_qs(parsed_url.query)
            if 'suspend' in queries and len(queries['suspend']) > 0 and queries['suspend'][0].lower() == 'true':
                suspend = True
            
            import pydevd; pydevd.settrace(host=hostname, port=port, suspend=suspend)
        except:
            PydevConnector._log.warn("Could not connect to PyDev remote debug server: %s" % (url))
    
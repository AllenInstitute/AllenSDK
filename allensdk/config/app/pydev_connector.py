# Copyright 2014 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

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
    
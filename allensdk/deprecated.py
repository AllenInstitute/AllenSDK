# Copyright 2015 Allen Institute for Brain Science
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


import warnings
import functools
from numpy import VisibleDeprecationWarning
# Python suppresses DeprecationWarning, so numpy made a UserWarning with a 
# different name.

    
def deprecated(message=None):

    if message is None:
        message = '' 
    
    def output_decorator(fn):
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
        
            warnings.warn("Function {0} is deprecated. {1}".format(
                          fn.__name__, message), 
                          category=VisibleDeprecationWarning, stacklevel=2)
            
            return fn(*args, **kwargs)
            
        return wrapper
        
    return output_decorator

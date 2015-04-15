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



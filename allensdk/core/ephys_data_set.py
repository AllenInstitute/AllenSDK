# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

class EphysDataSet( object ):
    def __init__(self, file_name):
        self.file_name = file_name
    
    def get_sweep(self, sweep_number):
        raise Exception("get_sweep not implemented")
    
    def set_sweep(self, sweep_number, stimulus, response):
        raise Exception("get_sweep not implemented")
    
    def get_spike_times(self, sweep_number):
        raise Exception("set_sweep_spike_times not implemented")
    
    def set_spike_times(self, sweep_number, spike_times):
        raise Exception("set_sweep_spike_times not implemented")

# Copyright 2016 Allen Institute for Brain Science
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

from allensdk.cam.CAM_NWB import CamNwbDataSet
class CamNwbLinter(object):
    def __init__(self):
        pass
    
    def do_lint(self, file_name):
        nwb = CamNwbDataSet(file_name)
        
        try:            
            nwb.get_fluorescence_traces()
            print('yay')
        except:
            print('no fluorescence_traces()')
            
        try:
            nwb.get_max_projection()
            print('yay')
        except:
            print('no max projection')
            
        try:            
            nwb.get_stimulus_table()
            print('yay')
        except:
            print('no stimulus table')
            
        try:
            nwb.get_stimulus_template()
            print('yay')
        except:
            print('no stimulus template')
            
        try:
            nwb.get_roi_mask()
            print('yay')
        except:
            print('no roi mask')
            
        try:
            nwb.get_meta_data()
            print('yay')
        except:
            print('no meta data')
            
        try:
            nwb.get_running_speed()
            print('yay')
        except:
            print('no running speed')
            
        try:
            nwb.get_motion_correction()
        except:
            print('no motion correction')
        
if __name__ == '__main__':
    cam_nwb_linter = CamNwbLinter()
    cam_nwb_linter.do_lint('/local1/cam_datasets/502382906/502382906.nwb')
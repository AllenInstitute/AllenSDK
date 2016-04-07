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

from allensdk.cam.cam_analysis import run_cam_analysis
import os

# params
lims_id = 502115959
depth = 175
stimulus = 'A'

# file names
save_dir = r'/Users/saskiad/Documents/Data/ophysdev/'
nwb_path = os.path.join(save_dir, "%d/%d.nwb" % (lims_id, lims_id))
save_path = os.path.join(save_dir, "%d/Data/%d.nwb" % (lims_id, lims_id))

# run the analysis
run_cam_analysis(stimulus, nwb_path, save_path, depth)


#     try:
#         (stimulus, nwb_path, save_path, lims_id) = sys.argv[-4:]
#         main(stimulus, nwb_path, save_path, lims_id)  
#         # A /local1/cam_datasets/501836392/501836392.nwb /local1/cam_datasets/501836392/Data 501836392 True        
#         # B /local1/cam_datasets/501886692/501886692.nwb /local1/cam_datasets/501886692/Data 501886692 True
#         # C /local1/cam_datasets/501717543/501717543.nwb /local1/cam_datasets/501717543/Data 501717543 True
#     except:
#         raise(Exception('please specify stimulus A, B or C, cam_directory, lims_id'))

#    main('A', '/Users/saskiad/Documents/Data/ophysdev/502115959/502115959.nwb', 'r/Users/saskiad/Documents/Data/ophysdev/502115959/Data', '502115959')        
#    main('B', '/local1/cam_datasets/501886692/501886692.nwb', '/local1/cam_datasets/501886692/Data', '501886692')
#    main('C', '/local1/cam_datasets/501717543/501717543.nwb', '/local1/cam_datasets/501717543/Data', '501717543')
    
#
##
####            

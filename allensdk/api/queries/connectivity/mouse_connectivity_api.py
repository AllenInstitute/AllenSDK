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

from allensdk.api.api import Api
import json

class MouseConnectivityApi(Api):
    '''http://help.brain-map.org/display/mouseconnectivity/API'''
    product_id = 5
    
    def __init__(self, base_uri=None):
        super(MouseConnectivityApi, self).__init__(base_uri)
    
    
    def build_query(self, structure_id=None, fmt='json'):
        '''Build the URL that will fetch meta data
        for all mouse connectivity projection experiments.
        
        Parameters
        ----------
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        
        if structure_id:
            structure_filter = '[id$eq%d]' % (structure_id)
        else:
            structure_filter = ''
        
        url = ''.join([self.rma_endpoint,
                       '/query.',
                       fmt,
                       '?q=',
                       'model::SectionDataSet',
                       ',rma::criteria,',
                       'products[id$eq%d]' % (MouseConnectivityApi.product_id),
                       ',rma::include,',
                       'specimen',
                       '(stereotaxic_injections',
                       '(primary_injection_structure,',
                       'structures',
                       structure_filter,
                       '))'])
        
        return url
    
    
    def build_detail_query(self, experiment_id, fmt='json'):
        '''
        
        Parameters
        ----------
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.rma_endpoint,
                       '/query.',
                       fmt,
                       '?q=',
                       'model::SectionDataSet',
                       ',rma::criteria,',
                       '[id$eq%d]' % (experiment_id),
                       ',rma::include,',
                       'specimen',
                       '(stereotaxic_injections',
                       '(primary_injection_structure,',
                       'structures,',
                       'stereotaxic_injection_coordinates)),',
                       'equalization,',
                       'sub_images',
                       ',rma::options',
                       "[order$eq'sub_images.section_number$asc']"])
        
        return url
    
    
    def build_projection_image_meta_info(self,
                                         experiment_id,
                                         section_number,
                                         fmt='json'):
        '''
        
        Parameters
        ----------
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.rma_endpoint,
                       '/query.',
                       fmt,
                       '?q=',
                       'model::SectionDataSet',
                       ',rma::criteria,',
                       '[id$eq%d]' % (experiment_id),
                       ',rma::include,',
                       'equalization,',
                       'sub_images',
                       '[section_number$eq%d]' % (section_number)])
        
        return url
    
    
    def read_response(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def get_experiments(self, structure_id):
        '''Retrieve the experimants data.'''
        data = self.do_query(self.build_query,
                             self.read_response,
                             structure_id)
        
        return data
    
    
    def get_experiment_detail(self, experiment_id):
        '''Retrieve the experimants data.'''
        data = self.do_query(self.build_detail_query,
                             self.read_response,
                             experiment_id)
        
        return data
    
    
    def get_projection_image_meta_info(self,
                                       experiment_id,
                                       section_number):
        '''Retrieve the experimants data.'''
        data = self.do_query(self.build_projection_image_meta_info,
                             self.read_response,
                             experiment_id,
                             section_number)
        
        return data
    
    
    def build_download_image_url(self,
                                 image_id,
                                 threshold_range,
                                 downsample=None,
                                 left=None,
                                 top=None,
                                 width=None,
                                 height=None):
        params = []
        
        if downsample:
            params.append('downsample=%d' % (downsample))
        
        if left:
            params.append('left=%d' % (left))
        
        if top:
            params.append('top=%d' % (top))
        
        if width:
            params.append('width=%d' % (width))
        
        if height:
            params.append('height=%d' % (height))
        
        params.append('range=%s' % (','.join([str(i) for i in threshold_range])))

        url = ''.join([self.section_image_download_endpoint,
                       '/',
                       str(image_id),
                       '?',
                       '&'.join(params)])
        
        return url
    
    
    def download_volumetric_data(self,
                                 data,
                                 file_name,
                                 voxel_resolution=None,
                                 save_file_path=None,
                                 release=None,
                                 coordinate_framework=None):
        '''
        Parameters
        ----------
        data : string
            'average_template', 'ara_nissl', 'annotation/ccf_2015', 'annotation/mouse_2011', or 'annotation/devmouse_2012'
        voxel_resolution : int
            10, 25, 50 or 100
        coordinate_framework : string
            'mouse_ccf' (default) or 'mouse_annotation'
        '''
        
        if voxel_resolution == None:
            voxel_resolution = 10
            
        if save_file_path == None:
            save_file_path = file_name
        
        if release == None:
            release = 'current-release'
        
        if coordinate_framework == None:
            coordinate_framework = 'mouse_ccf'
        
        url = ''.join([self.informatics_archive_endpoint,
                       '/%s/%s/' % (release, coordinate_framework),
                       data,
                       '/',
                       file_name])
        
        self.retrieve_file_over_http(url, save_file_path)


if __name__ == '__main__':
    import nrrd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    a = MouseConnectivityApi()
    #print(json.dumps(a.get_experiments()))
    #print(json.dumps(a.get_experiments(structure_id=385)))
    #print(json.dumps(a.get_experiment_detail(experiment_id=126862385)))
    #print(json.dumps(a.get_projection_image_meta_info(experiment_id=126862385,
    #                                                  section_number=74)))
    #a.download_volumetric_data('average_template', 'average_template_25.nrrd')
    #a.download_volumetric_data('ara_nissl', 'ara_nissl_25.nrrd')
    #a.download_volumetric_data('annotation/ccf_2015', 'annotation_25.nrrd')
    AVGT, metaAVGT = nrrd.read('average_template_25.nrrd')
    NISSL, metaNISSL = nrrd.read('ara_nissl_25.nrrd')
    ANO, metaANO = nrrd.read('annotation_25.nrrd')
    
    # Save one coronal section as PNG
    slice = AVGT[264,:,:].astype(float)
    slice /= np.max(slice)
    im = Image.fromarray(np.uint8(plt.cm.gray(slice)*255))
    im.save('output/avgt_coronal.png')
    
    slice = NISSL[264,:,:].astype(float)
    slice /= np.max(slice)
    im = Image.fromarray(np.uint8(plt.cm.gray(slice)*255))
    im.save('output/nissl_coronal.png')
    
    slice = ANO[264,:,:].astype(float)
    slice /= 2000
    im = Image.fromarray(np.uint8(plt.cm.jet(slice)*255))
    im.save('output/ano_coronal.png')
    
    # Save one sagittal section as PNG
    slice = AVGT[:,:,220].astype(float)
    slice /= np.max(slice)
    im = Image.fromarray(np.uint8(plt.cm.gray(slice)*255))
    im.save('output/avgt_sagittal.png')
    
    slice = NISSL[:,:,220].astype(float)
    slice /= np.max(slice)
    im = Image.fromarray(np.uint8(plt.cm.gray(slice)*255))
    im.save('output/nissl_sagittal.png')
    
    slice = ANO[:,:,220].astype(float)
    slice /= 2000
    im = Image.fromarray(np.uint8(plt.cm.jet(slice)*255))
    im.save('output/ano_sagittal.png')

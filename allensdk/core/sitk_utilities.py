# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2015-2018. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import SimpleITK as sitk
import numpy as np


def get_sitk_image_information(image):
    '''
    '''

    return {'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'size': image.GetSize(),
            'direction': image.GetDirection(),
            'ncomponents': image.GetNumberOfComponentsPerPixel()}


def set_sitk_image_information(image, information):
    '''
    '''

    if 'spacing' in information:
        image.SetSpacing(information['spacing'])
    if 'origin' in information:
        image.SetOrigin(information['origin'])
    if 'direction' in information:
        image.SetDirection(information['direction'])


def fix_array_dimensions(array, ncomponents=1):
    ''' Convenience function that reorders ndarray dimensions for io with SimpleITK

    Parameters
    ----------
    array : np.ndarray
        The array to be reordered
    ncomponents : int, optional
        Number of components per pixel, default 1. 

    Returns
    -------
    np.ndarray : 
        Reordered array

    '''

    act_size = list(array.shape)
    ndims = len(act_size)
    multicomponent = ncomponents > 1

    from_order = list(range( ndims - multicomponent ))
    to_order = list(range( ndims - multicomponent ))[::-1]

    if multicomponent:
        from_order += [-1]
        to_order += [-1]

    return np.ascontiguousarray(np.moveaxis(array, from_order, to_order))


def read_ndarray_with_sitk(path):
    '''
    '''

    image = sitk.ReadImage(str(path))
    information = get_sitk_image_information(image)
    image = sitk.GetArrayFromImage(image)

    image = fix_array_dimensions(image, information['ncomponents'])
    return image, information


def write_ndarray_with_sitk(array, path, **information):
    '''
    '''

    if not 'ncomponents' in information:
        information['ncomponents'] = 1
    ncomponents = information.pop('ncomponents')

    array = fix_array_dimensions(array, ncomponents)
    
    array = sitk.GetImageFromArray(array, ncomponents > 1)
    set_sitk_image_information(array, information)

    sitk.WriteImage(array, str(path))
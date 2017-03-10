# Copyright 2017 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division, print_function, absolute_import
from collections import defaultdict
import operator as op
import functools
import os

from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import numpy as np
import nrrd

from allensdk.core.structure_tree import StructureTree


class ReferenceSpace(object):

    @property
    def direct_voxel_map(self):
        if not hasattr(self, '_direct_voxel_map'):
            self.direct_voxel_counts()
        return self._direct_voxel_map 
        
    @direct_voxel_map.setter
    def direct_voxel_map(self, data):
        self._direct_voxel_map = data
    
    @property
    def total_voxel_map(self):
        if not hasattr(self, '_total_voxel_map'):
            self.total_voxel_counts()
        return self._total_voxel_map
        
    @total_voxel_map.setter
    def total_voxel_map(self, data):
        self._total_voxel_map = data
        
    def __init__(self, structure_tree, annotation, resolution):
        '''Handles brain structures in a 3d reference space
        
        Parameters
        ----------
        structure_tree : StructureTree
            Defines the heirarchy and properties of the brain structures.
        annotation : numpy ndarray
            3d volume whose elements are structure ids.
        resolution : length-3 tuple of numeric
            Resolution of annotation voxels along each dimension.
        
        '''
        
        self.structure_tree = structure_tree
        self.resolution = resolution
        
        self.annotation = np.ascontiguousarray(annotation)
        
    def direct_voxel_counts(self):
        '''Determines the number of voxels directly assigned to one or more 
        structures.
        
        Returns
        -------
        dict : 
            Keys are structure ids, values are the number of voxels directly 
            assigned to those structures.
        
        '''

        uniques = np.unique(self.annotation, return_counts=True)
        found = {k: v for k, v in zip(*uniques) if k != 0}

        self._direct_voxel_map = {k: (found[k] if k in found else 0) for k 
                                  in self.structure_tree.node_ids()}
          
    def total_voxel_counts(self):
        '''Determines the number of voxels assigned to a structure or its 
        descendants
            
        Returns
        -------
        dict : 
            Keys are structure ids, values are the number of voxels assigned 
            to structures' descendants.
        
        ''' 

        self._total_voxel_map = {}
        for stid in self.structure_tree.node_ids():

            desc_ids = self.structure_tree.descendant_ids([stid])[0]
            self._total_voxel_map[stid] = sum([self.direct_voxel_map[dscid] 
                                               for dscid in desc_ids])
    
    def remove_unassigned(self, update_self=True):
        '''Obtains a structure tree consisting only of structures that have 
        at least one voxel in the annotation.
        
        Parameters
        ----------
        update_self : bool, optional
            If True, the contained structure tree will be replaced,
            
        Returns
        -------
        list of dict : 
            elements are filtered structures
        
        '''
    
        structures = self.structure_tree.filter_nodes(
            lambda x: self.total_voxel_map[x['id']] > 0)
            
        if update_self:
            self.structure_tree = StructureTree(structures)
            
        return structures
    
    def make_structure_mask(self, structure_ids, direct_only=False):
        '''Return an indicator array for one or more structures
        
        Parameters
        ----------
        structure_ids : list of int
            Make a mask that indicates the union of these structures' voxels
        direct_only : bool, optional
            If True, only include voxels directly assigned to a structure in 
            the mask. Otherwise include voxels assigned to descendants.
            
        Returns
        -------
        numpy ndarray :
            Same shape as annotation. 1 inside mask, 0 outside.
        
        '''
    
        if direct_only:
            mask = np.zeros(self.annotation.shape, dtype=np.uint8, order='C')
            for stid in structure_ids:
                
                if self.direct_voxel_map[stid] == 0:
                    continue
                    
                mask[self.annotation == stid] = True
                
            return mask
            
        else:
            structure_ids = self.structure_tree.descendant_ids(structure_ids)
            structure_ids = set(reduce(op.add, structure_ids))
            return self.make_structure_mask(structure_ids, direct_only=True)
                        
    def many_structure_masks(self, structure_ids, output_cb=None, 
                             direct_only=False):
        '''Build one or more structure masks and do something with them
        
        Parameters
        ----------
        structure_ids : list of int
            Specify structures to be masked
        output_cb : function, optional
            Must have the following signature: output_cb(structure_id, fn). 
            On each requested id, fn will be curried to make a mask for that 
            id. Defaults to returning the structure id and mask.
        direct_only : bool, optional
            If True, only include voxels directly assigned to a structure in 
            the mask. Otherwise include voxels assigned to descendants.
            
        Yields
        -------
        Return values of output_cb called on each structure_id, structure_mask 
        pair.
        
        Notes
        -----
        output_cb is called on every yield, so any side-effects (such as 
        writing to a file) will be carried out regardless of what you do with 
        the return values. You do actually have to iterate through the output, 
        though.
        
        '''
        
        if output_cb is None:
            output_cb = ReferenceSpace.return_mask_cb
                                              
        for stid in structure_ids:
            yield output_cb(stid, functools.partial(self.make_structure_mask, 
                                                    [stid], direct_only))


    def check_coverage(self, structure_ids, domain_mask):
        '''Determines whether a spatial domain is completely covered by 
        structures in a set.
        
        Parameters
        ----------
        structure_ids : list of int 
            Specifies the set of structures to check.
        domain_mask : numpy ndarray
            Same shape as annotation. 1 inside the mask, 0 out. Specifies 
            spatial domain.
            
        Returns
        -------
        numpy ndarray : 
            1 where voxels are missing from the candidate, 0 where the 
            candidate exceeds the domain
        
        '''
    
        candidate_mask = self.make_structure_mask(structure_ids)
        return domain_mask - candidate_mask
        
    def validate_structures(self, structure_ids, domain_mask):
        '''Determines whether a set of structures produces an exact and 
        nonoverlapping tiling of a spatial domain
        
        Parameters
        ----------
        structure_ids : list of int 
            Specifies the set of structures to check.
        domain_mask : numpy ndarray
           Same shape as annotation. 1 inside the mask, 0 out. Specifies 
           spatial domain.
           
        Returns
        -------
        set : 
            Ids of structures that are the ancestors of other structures in 
            the supplied set.
        numpy ndarray : 
            Indicator for missing voxels.
            
        '''
        
        return [self.structure_tree.has_overlaps(structure_ids), 
                self.check_coverage(structure_ids, domain_mask)]
        
        
    def downsample(self, target_resolution):
        '''Obtain a smaller reference space by downsampling
        
        Parameters
        ----------
        target_resolution : tuple of numeric
            Resolution in microns of the output space.
        interpolator : string
            Method used to interpolate the volume. Currently only 'nearest' 
            is supported
            
        Returns
        -------
        ReferenceSpace : 
            A new ReferenceSpace with the same structure tree and a 
            downsampled annotation.
        
        '''
        
        factors = [ float(ii / jj) for ii, jj in zip(self.resolution, 
                                                     target_resolution)]
                                                     
        target = zoom(self.annotation, factors, order=0)
        
        return ReferenceSpace(self.structure_tree, target, target_resolution)
        
        
    def get_slice_image(self, axis, position, cmap=None):
        '''Produce a AxBx3 RGB image from a slice in the annotation
        
        Parameters
        ----------
        axis : int
            Along which to slice the annotation volume. 0 is coronal, 1 is 
            horizontal, and 2 is sagittal.
        position : int 
            In microns. Take the slice from this far along the specified axis.
        cmap : dict, optional
            Keys are structure ids, values are rgb triplets. Defaults to 
            structure color_hex_triplets. 
            
        Returns
        -------
        np.ndarray : 
            RGB image array. 
            
        Notes
        -----
        If you assign a custom colormap, make sure that you take care of the 
        background in addition to the structures.
        
        '''
        
        if cmap is None:
            cmap = self.structure_tree.get_colormap()
            cmap[0] = [0, 0, 0]
        
        position = int(np.around(position / self.resolution[axis]))
        image = np.squeeze(self.annotation.take([position], axis=axis))
            
        return np.reshape([cmap[point] for point in image.flat], 
                      list(image.shape) + [3]).astype(np.uint8)
            
            
    @staticmethod
    def return_mask_cb(structure_id, fn):
        '''A basic callback for many_structure_masks
        '''
    
        return structure_id, fn()
        
        
    @staticmethod
    def check_and_write(base_dir, structure_id, fn):
        '''A many_structure_masks callback that writes the mask to a nrrd file 
        if the file does not already exist.
        '''
    
        mask_path = os.path.join(base_dir, 
                                 'structure_{0}.nrrd'.format(structure_id))
        
        if not os.path.exists(mask_path):
            nrrd.write(mask_path, fn())
            
        return structure_id


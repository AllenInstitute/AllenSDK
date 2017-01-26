from __future__ import division, print_function, absolute_import
from collections import defaultdict
import operator as op

from scipy.misc import imresize
import numpy as np

from allensdk.core.structure_tree import StructureTree


class ReferenceSpace(object):


    @property
    def direct_voxel_map(self):
        structure_ids = self.structure_tree.node_ids()
        return self.direct_voxel_counts(structure_ids)
    
    
    @property
    def total_voxel_map(self):
        structure_ids = self.structure_tree.node_ids()
        return self.total_voxel_counts(structure_ids)


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
        
        self._direct_voxel_map = defaultdict(lambda *x: None, {})
        self._total_voxel_map = defaultdict(lambda *x: None, {})

        
    def direct_voxel_counts(self, structure_ids):
        '''Determines the number of voxels directly assigned to one or more 
        structures.
        
        Parameters
        ----------
        structure_ids : list of int
        
        
        Returns
        -------
        dict : 
            Keys are structure ids, values are the number of voxels directly 
            assigned to input structures.
        
        '''
    
        counts = {}
        for stid in structure_ids:
        
            counts[stid] = self._direct_voxel_map[stid]
        
            if counts[stid] is None:
                counts[stid] = (self.annotation.size \
                                - np.count_nonzero(self.annotation - stid))
                                
        self._direct_voxel_map.update(counts)
        return counts
        
        
    def total_voxel_counts(self, structure_ids):
        '''Determines the number of voxels assigned to a structure or its 
        descendants
        
        Parameters
        ----------
        structure_ids : list of int
            Get counts for these structures.
            
        Returns
        -------
        dict : 
            Keys are structure ids, values are the number of voxels assigned 
            to input structures' descendants.
        
        ''' 
        

        counts = {}
        for stid in structure_ids:
        
            counts[stid] = self._total_voxel_map[stid]
        
            if counts[stid] is None:
                desc_ids = self.structure_tree.descendant_ids([stid])[0]
                counts[stid] = sum(self.direct_voxel_counts(desc_ids).values())
                                
        self._total_voxel_map.update(counts)
        return counts
    
    
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
            mask = np.zeros(self.annotation.shape, dtype=np.bool_, order='C')
            for stid in structure_ids:
                
                print(stid)
                if self.direct_voxel_counts([stid]) == 0:
                    continue
                    
                mask[np.where(self.annotation == stid)] = True
                
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
        output_cb : function | structure_id, structure_mask => ?
            Will be applied to each id and mask. The default just gives you 
            structure mask ndarrays.
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
            output_cb = lambda structure_id, structure_mask: structure_mask
        
        for stid in structure_ids:
            yield output_cb(stid, self.make_structure_mask([stid], 
                            direct_only))

        
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
            Indicator for missing voxels.
        
        '''
    
        candidate_mask = self.make_structure_mask(structure_ids)
        return np.logical_and(domain_mask, np.logical_not(candidate_mask))
        
        
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
            
        Returns
        -------
        ReferenceSpace : 
            A new ReferenceSpace with the same structure tree and a 
            downsampled annotation.
        
        '''
        
        # divide to go from source -> target
        factors = [ ii / jj for ii, jj in zip(self.resolution, 
                                              target_resolution)]
        target_size = [int(np.around(sc * shp)) for sc, shp 
                       in zip(scale, self.annotation.shape)]
    
        target_annotation = imresize(self.annotation, target_size, 
                                     interp='nearest')
                                                
        return ReferenceSpace(self.structure_tree, target_annotation, 
                              target_resolution)
        

    
# An example for many_structure_masks. Use by currying  a la:
# from functools import partial
# output_cb = functools.partial(write_nrrd_cb, my_out_dir, my_iso_res)
def write_nrrd_cb(output_directory, isometric_resolution, structure_id, 
                  structure_mask):
    path = os.path.join(output_directory, 
                        'structure_mask_{0}_{1}'.format(structure_id, 
                                                        isometric_resolution))
    nrrd.write(path, structure_mask)


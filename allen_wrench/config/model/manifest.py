# Copyright 2014 Allen Institute for Brain Science
# Licensed under the Allen Institute Terms of Use (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.alleninstitute.org/Media/policies/terms_of_use_content.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import logging

class Manifest(object):
    """Manages the location of external files and large objects 
     referenced in an Allen Wrench configuration """

    DIR = 'dir'
    FILE = 'file'
    DIRNAME = 'dir_name'
    log = logging.getLogger(__name__)

    def __init__(self, config=None):
        self.path_info = {}
        
        if config != None:
            self.load_config(config)
       
        
    def load_config(self, config):
        ''' Load paths into the manifest from an Allen Wrench config section
            :param config: the manifest section of an Allen Wrench config
            :type config: Config
        '''
        for path_info in config:
            path_type = path_info['type']
            path_format = None
            path_schema = None
            if 'format' in path_info:
                path_format = path_info['format']
                
            if 'schema' in path_info:
                path_schema = path_info['schema']
            
            if path_type == 'file':
                try:
                    parent_key = path_info['parent_key']
                except:
                    parent_key = None
                
                self.add_file(path_info['key'],
                              path_info['spec'],
                              parent_key,
                              path_format,
                              path_schema)
            elif path_type == 'dir':
                try:
                    parent_key = path_info['parent_key']
                except:
                    parent_key = None
                
                spec = path_info['spec']
                absolute = False
                if spec[0] == '/':
                    absolute = True
                self.add_path(path_info['key'],
                              path_info['spec'],
                              path_type,
                              absolute,
                              path_format,
                              parent_key)
            else:
                Manifest.log.warning("Unknown path type in manifest: %s" %
                                     (path_type))
                        
        
    def add_path(self, key, path, path_type=DIR,
                 absolute=True, path_format=None, parent_key=None):
        if parent_key:
            path_args = []
            
            try:
                parent_path = self.path_info[parent_key]['spec']
                path_args.append(parent_path)
            except:
                Manifest.log.error("cannot resolve directory key %s" % (parent_key))
                raise
            path_args.extend(path.split('/'))
            path = os.path.join(*path_args)
        
        # TODO: relative paths need to be considered better
        if absolute == True:
            path = os.path.abspath(path)
        else:
            path = os.path.abspath(path)
            
        if path_type == Manifest.DIRNAME:
            path = os.path.dirname(path)
            
        self.path_info[key] = { 'type': path_type,
                                'spec': path}
        
        if path_type == Manifest.FILE:
            self.path_info[key]['format'] = path_format


    def add_paths(self, path_info):
        ''' add information about paths stored in the manifest.
            :param path_info: a dictionary with information about the new paths
            :type path_info: dict
        '''
        for path_key, path_data in path_info.items():
            path_format = None
            
            if 'format' in path_data:
                path_format = path_data['format']

            if 'schema' in path_data:
                path_schema = path_data['schema']
            else:
                path_schema = { 'data': [] }                
                            
            Manifest.log.info("Adding path.  type: %s, format: %s, spec: %s" %
                              (path_data['type'],
                               path_data['spec'],
                               path_format))
            self.path_info[path_key] = { 'type': path_data['type'],
                                         'spec': path_data['spec'],
                                         'format': path_format,
                                         'schema': path_schema }
    
    
    def add_file(self,
                 file_key,
                 file_name,
                 dir_key=None,
                 path_format=None,
                 schema=None):
        path_args = []
        
        if dir_key:
            try:
                dir_path = self.path_info[dir_key]['spec']
                path_args.append(dir_path)
            except:
                Manifest.log.error("cannot resolve directory key %s" % (dir_key))
                raise
        else:
            path_args.append(os.curdir)
        
        path_args.extend(file_name.split('/'))
        file_path = os.path.join(*path_args)
        
        self.path_info[file_key] = { 'type': Manifest.FILE,
                                     'format': path_format,
                                     'spec': file_path }
        
        if schema != None:
            self.path_info[file_key].update({ 'schema': schema })

    def get_path(self, path_key, *args):
        path_spec = str(self.path_info[path_key]['spec'].encode('ascii',
                                                                'ignore'))
        
        if args != None and len(args) != 0:
            path = path_spec % args
        else:
            path = path_spec
            
        return path
    
    def get_format(self, path_key):
        path_entry = self.path_info[path_key]
        path_format = None
        
        if 'format' in path_entry:
            path_format = path_entry['format']
            
        return path_format
    
    
    def get_schema(self, path_key):
        path_entry = self.path_info[path_key]
        path_schema = None

        if 'schema' in path_entry:
            path_schema = path_entry['schema']
            
        return path_schema
        
    
    def create_dir(self, path_key):
        dir_path = self.get_path(path_key)
        
        try:
            os.stat(dir_path)
        except:
            os.mkdir(dir_path)
            
    def check_dir(self, path_key, do_exit=False):
        dir_path = self.get_path(path_key)
        
        if not os.path.exists(dir_path):
            Manifest.log.fatal('Directory %s does not exist; exiting.' % 
                               (dir_path))
            if do_exit == True:
                quit()
    
    def resolve_paths(self, description_dict, suffix='_key'):
        key_pattern =  re.compile('(.*)%s$' % (suffix))
        
        for description_key, manifest_key in description_dict.items():
            m = key_pattern.match(description_key)
            if m:
                real_key = m.group(1)  # i.e. job_dir_key -> job_dir
                filename = self.get_path(manifest_key)
                description_dict[real_key] = filename
                del description_dict[description_key]

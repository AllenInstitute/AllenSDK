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

import unittest
from mock import MagicMock
from allensdk.config.manifest_builder import ManifestBuilder


class ManifestTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ManifestTests, self).__init__(*args, **kwargs)
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testManifestConstructor(self):
        builder = ManifestBuilder()
        builder.add_path('BASEDIR', '/home/username/example')
        
        manifest = builder.get_manifest()
                
        expected = '/home/username/example'
        actual = manifest.get_path('BASEDIR')
        self.assertEqual(expected, actual)

    def testManifestParent(self):
        builder = ManifestBuilder()
        builder.add_path('BASEDIR',
                         '/home/username/example')
        builder.add_path('WORKDIR',
                         'work',
                         parent_key='BASEDIR')
        
        manifest = builder.get_manifest()
                
        expected = '/home/username/example/work'
        actual = manifest.get_path('WORKDIR')
        self.assertEqual(expected, actual)


    def testManifestBuilderDataFrame(self):
        builder = ManifestBuilder()
        builder.add_path('BASEDIR',
                         '/home/username/example')
        builder.add_path('WORKDIR',
                         'work',
                         parent_key='BASEDIR')
        
        builder_df = builder.as_dataframe()
        
        self.assertTrue('key' in builder_df.keys())
        self.assertTrue('type' in builder_df.keys())
        self.assertTrue('spec' in builder_df.keys())
        self.assertTrue('parent_key' in builder_df.keys())
        self.assertTrue('format' in builder_df.keys())        
        self.assertEqual(5, len(builder_df.keys()))


    def testManifestDataFrame(self):
        builder = ManifestBuilder()
        builder.add_path('BASEDIR',
                         '/home/username/example')
        builder.add_path('WORKDIR',
                         'work',
                         parent_key='BASEDIR')
        
        manifest = builder.get_manifest()
        df = manifest.as_dataframe()
        
        self.assertTrue('type' in df.keys())
        self.assertTrue('spec' in df.keys())
        self.assertEqual(2, len(df.keys()))
        

        

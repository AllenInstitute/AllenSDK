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


import pytest
from allensdk.config.manifest_builder import ManifestBuilder


@pytest.fixture
def builder():
    b = ManifestBuilder()
    b.add_path('BASEDIR', '/home/username/example')

    return b


def testManifestConstructor(builder):
    manifest = builder.get_manifest()
    expected = '/home/username/example'
    actual = manifest.get_path('BASEDIR')
    assert(expected == actual)


def testManifestParent(builder):
    builder.add_path('WORKDIR',
                     'work',
                     parent_key='BASEDIR')
    manifest = builder.get_manifest()
    expected = '/home/username/example/work'
    actual = manifest.get_path('WORKDIR')
    assert(expected == actual)


def testManifestBuilderDataFrame(builder):
    builder.add_path('WORKDIR',
                     'work',
                     parent_key='BASEDIR')
    builder_df = builder.as_dataframe()

    assert('key' in builder_df.keys())
    assert('type' in builder_df.keys())
    assert('spec' in builder_df.keys())
    assert('parent_key' in builder_df.keys())
    assert('format' in builder_df.keys())
    assert(5 == len(builder_df.keys()))


def testManifestDataFrame(builder):
    builder.add_path('WORKDIR',
                     'work',
                     parent_key='BASEDIR')

    manifest = builder.get_manifest()
    df = manifest.as_dataframe()

    assert('type' in df.keys())
    assert('spec' in df.keys())
    assert(2 == len(df.keys()))

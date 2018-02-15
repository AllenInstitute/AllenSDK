# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
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
import pytest
import os
from allensdk.config.manifest_builder import ManifestBuilder
from allensdk.config.manifest import Manifest


@pytest.fixture
def builder():
    b = ManifestBuilder()
    b.add_path('BASEDIR', '/home/username/example')

    return b


def testManifestConstructor(builder):
    manifest = builder.get_manifest()
    expected = os.path.abspath('/home/username/example')
    actual = manifest.get_path('BASEDIR')
    assert(expected == actual)


def testManifestParent(builder):
    builder.add_path('WORKDIR',
                     'work',
                     parent_key='BASEDIR')
    manifest = builder.get_manifest()
    expected = os.path.abspath('/home/username/example/work')
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


def safe_mkdir_root_dir():
    directory = os.path.abspath(os.sep)
    Manifest.safe_mkdir(directory) # should not error
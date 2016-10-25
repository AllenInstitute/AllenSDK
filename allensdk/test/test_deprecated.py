# Copyright 2016 Allen Institute for Brain Science
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
import pytest
from allensdk.deprecated import deprecated
import warnings


@pytest.fixture
def deprecated_method():

    @deprecated
    def i_am_deprecated():
        pass

    return i_am_deprecated


def test_deprecated(deprecated_method):
    expected = "Function i_am_deprecated is deprecated."

    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter('always')
        deprecated_method()

        assert expected == str(c[-1].message)

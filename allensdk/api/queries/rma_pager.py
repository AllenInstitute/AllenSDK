# Copyright 2017 Allen Institute for Brain Science
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
import functools


class RmaPager(object):
    def __init__(self):
        pass

    @staticmethod
    def pager(fn,
              *args,
              **kwargs):
        total_rows = kwargs.pop('total_rows', None)
        num_rows = kwargs.get('num_rows', None)

        if total_rows == 'all':
            start_row = 0
            result_count = num_rows
            kwargs = kwargs
            kwargs['count'] = False

            while result_count == num_rows:
                kwargs['start_row'] = start_row
                data = fn(*args, **kwargs)
                start_row = start_row + num_rows
                result_count = len(data)
                for r in data:
                    yield r
        else:
            start_row = 0
            kwargs = kwargs
            kwargs['count'] = False

            while start_row < total_rows:
                kwargs['start_row'] = start_row

                data = fn(*args, **kwargs)
                result_count = len(data)
                start_row = start_row + result_count
                for r in data:
                    yield r

def pageable(total_rows=None,
             num_rows=None):
    def decor(func):
        decor.total_rows=total_rows
        decor.num_rows=num_rows

        @functools.wraps(func)
        def w(*args,
              **kwargs):
            if decor.num_rows and not 'num_rows' in kwargs:
                kwargs['num_rows'] = decor.num_rows
            if decor.total_rows and not 'total_rows' in kwargs:
                kwargs['total_rows'] = decor.total_rows

            result = RmaPager.pager(func,
                                    *args,
                                    **kwargs)
            return result
        return w
    return decor

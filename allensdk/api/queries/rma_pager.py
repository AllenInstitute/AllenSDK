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
        pages = kwargs.pop('pages', None)
        total_rows = kwargs.pop('total_rows', None)
        num_rows = kwargs.get('num_rows', None)

        def single_call(s):
            kwargs['start_row'] = s
            kwargs['count'] = False

            data = fn(*args, **kwargs)
            return data

        return functools.reduce( (lambda d, s: d.extend(single_call(s)) or d),
                                 range(0, total_rows, num_rows),
                                 [])


def pageable(pages=None,
             total_rows=None,
             num_rows=None):
    def decor(func):
        decor.total_rows=total_rows
        decor.pages=pages
        decor.num_rows=num_rows

        @functools.wraps(func)
        def w(*args,
              **kwargs):
            if decor.pages and not 'pages' in kwargs:
                kwargs['pages'] = decor.pages
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
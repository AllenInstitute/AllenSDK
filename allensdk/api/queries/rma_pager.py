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

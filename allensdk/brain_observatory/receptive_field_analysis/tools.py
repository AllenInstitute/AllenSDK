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

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    return {key:[item[key] for item in list_of_dicts] for key in list_of_dicts[0].keys() }

def dict_generator(indict, pre=None):

    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key] ):
                    yield d
            elif isinstance(value, list):
                for v in value:
                    for d in dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield indict

def read_h5_group(g):
    return_dict = {}
    if len(g.attrs) > 0:
        return_dict['attrs'] = dict(g.attrs)
    for key in g:
        if key == 'data':
            return_dict[key] = g[key].value
        else:
            return_dict[key] = read_h5_group(g[key])

    return return_dict


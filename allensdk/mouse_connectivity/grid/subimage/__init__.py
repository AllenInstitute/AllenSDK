from .classic_count_subimage import ClassicCountSubImage
from .cav_subimage import CavSubImage
from .classic_subimage import ClassicSubImage


def get_subimage_class(case):
    if case == 'classic':
        return ClassicSubImage
    elif case == 'classic_count':
        return ClassicCountSubImage
    elif case == 'cav':
        return CavSubImage
    else:
        raise ValueError('unrecognized case: {}'.format(case))
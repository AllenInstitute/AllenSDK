
from .writers import classic_writer, count_writer, cav_writer
from .subimage import CavSubimage, CountSubimage, ClassicSubimage


cases = {
    'classic': {
        'writer': classic_writer,
        'subimage': ClassicSubimage
    },
    'count': {
        'writer': count_writer,
        'subimage': CountSubimage
    },
    'cav': {
        'writer': cav_writer,
        'subimage': CavSubimage
    }
}
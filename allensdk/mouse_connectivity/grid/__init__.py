
from .subimage import CavSubImage, ClassicSubImage, CountSubImage
from .writers import cav_writer, classic_writer, count_writer

cases = {
    'classic': {
        'writer': classic_writer,
        'subimage': ClassicSubImage
    },
    'count': {
        'writer': count_writer,
        'subimage': CountSubImage
    },
    'cav': {
        'writer': cav_writer,
        'subimage': CavSubImage
    }
}
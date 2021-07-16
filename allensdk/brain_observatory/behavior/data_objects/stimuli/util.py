from pathlib import Path


def convert_filepath_caseinsensitive(filename_in):
    return filename_in.replace('TRAINING', 'training')


def get_image_set_name(image_set_path: str) -> str:
    """
    Strips the stem from the image_set filename
    """
    return Path(image_set_path).stem
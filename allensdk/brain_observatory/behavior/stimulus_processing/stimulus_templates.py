from typing import Dict, List

import numpy as np

from allensdk.brain_observatory.behavior.stimulus_processing.util import \
    get_image_set_name, convert_filepath_caseinsensitive


class StimulusImage:
    """Container class for image stimuli"""
    def __init__(self, name: str, unwarped: np.ndarray):
        """
        Parameters
        ----------
        name:
            Name of the image
        unwarped
            The unwarped image data
        """
        self._name = name
        self._unwarped = unwarped

    @property
    def name(self):
        return self._name

    @property
    def unwarped(self):
        return self._unwarped

    def __getitem__(self, item: str) -> np.ndarray:
        """
        Takes "unwarped" as input and returns the unwarped array
        """
        return self._unwarped

    def __repr__(self):
        d = {
            'unwarped': f'numpy array {self._unwarped.shape}'
        }
        return f'{d}'


class StimulusTemplates:
    """Container class for a collection of image stimuli"""
    def __init__(self, image_set_filepath: str, image_attributes: List[dict],
                 images: List[np.ndarray]):
        """
        Parameters
        ----------
        image_set_filepath:
            the path to the image set
        image_attributes
            List of image attributes as returned by the stimulus pkl
        images
            List of images as returned by the stimulus pkl
        """
        image_set_name = get_image_set_name(image_set_path=image_set_filepath)
        self._image_set_name = image_set_name

        image_set_filepath = convert_filepath_caseinsensitive(
            image_set_filepath)
        self._image_set_filepath = image_set_filepath

        self._images: Dict[str, StimulusImage] = {}

        for attr, image in zip(image_attributes, images):
            image_name = attr['image_name']
            self.__add_image(name=image_name, unwarped=image)

    @property
    def image_set_name(self):
        return self._image_set_name

    def keys(self):
        return self._images.keys()

    def values(self):
        return self._images.values()

    def items(self):
        return self._images.items()

    def __add_image(self, name, unwarped: np.ndarray):
        """
        Parameters
        ----------
        name:
            Name of the image
        unwarped
            The unwarped image array
        """
        image = StimulusImage(name=name, unwarped=unwarped)
        self._images[name] = image

    def __getitem__(self, item) -> StimulusImage:
        """
        Given an image name, returns the corresponding StimulusImage
        """
        return self._images[item]

    def __len__(self):
        return len(self._images)

    def __iter__(self):
        yield from self._images

    def __repr__(self):
        return f'{self._images}'

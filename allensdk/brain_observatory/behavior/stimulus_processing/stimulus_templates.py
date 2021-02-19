from typing import Dict, List
import warnings

import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.stimulus_processing.util import \
    convert_filepath_caseinsensitive


class StimulusImage(np.ndarray):
    """Container class for image stimuli"""
    def __new__(cls, input_array: np.ndarray, name: str):
        """
        Parameters
        ----------
        name:
            Name of the image
        values
            The unwarped image values
        """
        obj = np.asarray(input_array).view(cls)
        obj._name = name
        return obj

    @property
    def name(self):
        return self._name


class StimulusTemplate:
    """Container class for a collection of image stimuli"""
    def __init__(self, image_set_name: str, image_attributes: List[dict],
                 images: List[np.ndarray]):
        """
        Parameters
        ----------
        image_set_name:
            the name of the image set
        image_attributes
            List of image attributes as returned by the stimulus pkl
        images
            List of images as returned by the stimulus pkl
        """
        self._image_set_name = image_set_name

        image_set_name = convert_filepath_caseinsensitive(
            image_set_name)
        self._image_set_filepath = image_set_name

        self._images: Dict[str, StimulusImage] = {}

        for attr, image in zip(image_attributes, images):
            image_name = attr['image_name']
            self.__add_image(name=image_name, values=image)

    @property
    def image_set_name(self) -> str:
        return self._image_set_name

    @property
    def image_names(self) -> List[str]:
        return list(self.keys())

    @property
    def images(self) -> List[StimulusImage]:
        return list(self.values())

    def keys(self):
        return self._images.keys()

    def values(self):
        return self._images.values()

    def items(self):
        return self._images.items()

    def to_dataframe(self) -> pd.DataFrame:
        index = pd.Index(self.image_names, name='image_name')
        df = pd.DataFrame({'image': self.images}, index=index)
        df.name = self._image_set_name
        return df

    def __add_image(self, name: str, values: np.ndarray):
        """
        Parameters
        ----------
        name:
            Name of the image
        values
            The unwarped image values
        """
        image = StimulusImage(input_array=values, name=name)
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

    def __eq__(self, other: object):
        if isinstance(other, StimulusTemplate):
            if self.image_set_name != other.image_set_name:
                return False

            if sorted(self.image_names) != sorted(other.image_names):
                return False

            for (img_name, self_img) in self.items():
                other_img = other._images[img_name]
                if not np.array_equal(self_img, other_img):
                    return False

            return True
        else:
            raise NotImplementedError(
                "Cannot compare a StimulusTemplate with an object of type: "
                f"{type(other)}!")

from typing import Dict, List

import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.stimulus_processing.util import \
    convert_filepath_caseinsensitive
from allensdk.brain_observatory.stimulus_info import BrainObservatoryMonitor


class StimulusImage:
    """Container class for image stimuli"""

    def __init__(self, warped: np.ndarray, unwarped: np.ndarray, name: str):
        """
        Parameters
        ----------
        warped:
            The warped stimulus image
        unwarped:
            The unwarped stimulus image
        name:
            Name of the stimulus image
        """
        self._name = name
        self.warped = warped
        self.unwarped = unwarped

    @property
    def name(self):
        return self._name


class StimulusImageFactory:
    """Factory for StimulusImage"""
    _monitor = BrainObservatoryMonitor()

    def from_pkl(self, input_array: np.ndarray, name: str) -> StimulusImage:
        """Creates a StimulusImage from pkl input.
        Image needs to be warped and preprocessed"""
        resized, unwarped = self._get_unwarped(arr=input_array)
        warped = self._get_warped(arr=resized)
        image = StimulusImage(name=name, warped=warped, unwarped=unwarped)
        return image

    @staticmethod
    def from_nwb(warped: np.ndarray, unwarped: np.ndarray,
                 name: str) -> StimulusImage:
        """Creates a StimulusImage from nwb input.
        Image has already been warped and preprocessed"""
        image = StimulusImage(name=name, warped=warped, unwarped=unwarped)
        return image

    def _get_warped(self, arr: np.ndarray):
        """Note: The Stimulus image is warped when shown to the mice to account
        "for distance of the flat screen to the eye at each point on
        the monitor."""
        return self._monitor.warp_image(img=arr)

    def _get_unwarped(self, arr: np.ndarray):
        """This produces the pixels that would be visible in the unwarped image
        post-warping"""
        # 1. Resize image to the same size as the monitor
        resized_array = self._monitor.natural_scene_image_to_screen(
            arr, origin='upper')
        # 2. Remove unseen pixels
        arr = self._exclude_unseen_pixels(arr=resized_array)

        return resized_array, arr

    def _exclude_unseen_pixels(self, arr: np.ndarray):
        """After warping, some pixels are not visible on the screen.
        This sets those pixels to nan to make downstream analysis easier."""
        mask = self._monitor.get_mask()
        arr = arr.astype(np.float)
        arr *= mask
        arr[arr == 0] = np.nan
        return arr

    def _warp(self, arr: np.ndarray) -> np.ndarray:
        """The Stimulus image is warped when shown to the mice to account
        "for distance of the flat screen to the eye at each point on
        the monitor." This applies the warping."""
        return self._monitor.warp_image(img=arr)


class StimulusTemplate:
    """Container class for a collection of image stimuli"""

    def __init__(self, image_set_name: str, images: List[StimulusImage]):
        """
        Parameters
        ----------
        image_set_name:
            the name of the image set
        images
            List of images
        """
        self._image_set_name = image_set_name

        image_set_name = convert_filepath_caseinsensitive(
            image_set_name)
        self._image_set_filepath = image_set_name

        self._images: Dict[str, StimulusImage] = {}

        for image in images:
            self._images[image.name] = image

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
        warped = [img.warped for img in self.images]
        unwarped = [img.unwarped for img in self.images]
        df = pd.DataFrame({'unwarped': unwarped, 'warped': warped},
                          index=index)
        df.name = self._image_set_name
        return df

    def __add_image(self, name: str, values: np.ndarray):
        """
        Parameters
        ----------
        name:
            Name of the image
        values
            The image values
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
                warped_equal = np.array_equal(
                    self_img.warped, other_img.warped)
                unwarped_equal = np.allclose(self_img.unwarped,
                                             other_img.unwarped,
                                             equal_nan=True)
                if not (warped_equal and unwarped_equal):
                    return False

            return True
        else:
            raise NotImplementedError(
                "Cannot compare a StimulusTemplate with an object of type: "
                f"{type(other)}!")


class StimulusTemplateFactory:
    """Factory for StimulusTemplate"""
    @staticmethod
    def from_pkl(image_set_name: str, image_attributes: List[dict],
                 images: List[np.ndarray]) -> StimulusTemplate:
        """Create StimulusTemplate from pkl input"""
        stimulus_images = []
        for i, image in enumerate(images):
            factory = StimulusImageFactory()
            name = image_attributes[i]['image_name']
            stimulus_image = factory.from_pkl(name=name, input_array=image)
            stimulus_images.append(stimulus_image)
        return StimulusTemplate(image_set_name=image_set_name,
                                images=stimulus_images)

    @staticmethod
    def from_nwb(image_set_name: str, image_attributes: List[dict],
                 unwarped: List[np.ndarray], warped: List[np.ndarray]) -> \
            StimulusTemplate:
        """Create StimulusTemplate from nwb input."""
        stimulus_images = []
        for i, attrs in enumerate(image_attributes):
            warped_image = warped[i]
            unwarped_image = unwarped[i]
            name = attrs['image_name']
            stimulus_image = StimulusImageFactory.from_nwb(
                name=name, warped=warped_image, unwarped=unwarped_image)
            stimulus_images.append(stimulus_image)
        return StimulusTemplate(image_set_name=image_set_name,
                                images=stimulus_images)

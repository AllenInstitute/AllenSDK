# All of the omitted stimuli have a duration of 250ms as defined
# by the Visual Behavior team. For questions about duration contact that
# team.
import inspect
import logging
import os
from typing import Union

from pynwb import NWBFile, ProcessingModule, NWBHDF5IO
from pynwb.base import Images
from pynwb.image import GrayscaleImage

from allensdk.brain_observatory.behavior.behavior_session import \
    BehaviorSession
from allensdk.brain_observatory.behavior.image_api import ImageApi, Image
from allensdk.brain_observatory.session_api_utils import sessions_are_equal
from allensdk.core import DataObject, JsonReadableInterface, \
    NwbReadableInterface, NwbWritableInterface


def get_column_name(table_cols: list,
                    possible_names: set) -> str:
    """
    This function returns a column name, given a table with unknown
    column names and a set of possible column names which are expected.
    The table column name returned should be the only name contained in
    the "expected" possible names.
    :param table_cols: the table columns to search for the possible name within
    :param possible_names: the names that could exist within the data columns
    :return: the first entry of the intersection between the possible names
             and the names of the columns of the stimulus table
    """

    column_set = set(table_cols)
    column_names = list(column_set.intersection(possible_names))
    if not len(column_names) == 1:
        raise KeyError("Table expected one name column in intersection, found:"
                       f" {column_names}")
    return column_names[0]


def get_image(nwbfile: NWBFile, name: str, module: str) -> Image:
    nwb_img = nwbfile.processing[module].get_data_interface('images')[name]
    data = nwb_img.data
    resolution = nwb_img.resolution  # px/cm
    spacing = [resolution * 10, resolution * 10]

    img = ImageApi.serialize(data, spacing, 'mm')
    img = ImageApi.deserialize(img=img)
    return img


def add_image_to_nwb(nwbfile: NWBFile, image_data: Image, image_name: str):
    """
    Adds image given by image_data with name image_name to nwbfile

    Parameters
    ----------
    nwbfile
        nwbfile to add image to
    image_data
        The image data
    image_name
        Image name

    Returns
    -------
    None
    """
    module_name = 'ophys'
    description = '{} image at pixels/cm resolution'.format(image_name)

    data, spacing, unit = image_data

    assert spacing[0] == spacing[1] and len(
        spacing) == 2 and unit == 'mm'

    if module_name not in nwbfile.processing:
        ophys_mod = ProcessingModule(module_name,
                                     'Ophys processing module')
        nwbfile.add_processing_module(ophys_mod)
    else:
        ophys_mod = nwbfile.processing[module_name]

    image = GrayscaleImage(image_name,
                           data,
                           resolution=spacing[0] / 10,
                           description=description)

    if 'images' not in ophys_mod.containers:
        images = Images(name='images')
        ophys_mod.add_data_interface(images)
    else:
        images = ophys_mod['images']
    images.add_image(image)


class NWBWriter:
    """Base class for writing NWB files"""
    def __init__(self,
                 nwb_filepath: str,
                 session_data: dict,
                 serializer: Union[
                     JsonReadableInterface,
                     NwbReadableInterface,
                     NwbWritableInterface]):
        """

        Parameters
        ----------
        nwb_filepath: path to write nwb
        session_data: dict representation of data to instantiate `serializer`
            and write nwb
        serializer: The class to use to read `session_data` and write nwb.
            Must implement `JsonReadableInterface`, `NwbReadableInterface`,
            `NwbWritableInterface`
        """
        self._serializer = serializer
        self._session_data = session_data
        self._nwb_filepath = nwb_filepath
        self.nwb_filepath_inprogress = nwb_filepath + '.inprogress'
        self._nwb_filepath_error = nwb_filepath + '.error'

        logging.basicConfig(
            format='%(asctime)s - %(process)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

        # Clean out files from previous runs:
        for filename in [self.nwb_filepath_inprogress,
                         self._nwb_filepath_error,
                         nwb_filepath]:
            if os.path.exists(filename):
                os.remove(filename)

    @property
    def nwb_filepath(self) -> str:
        """Path to write nwb file"""
        return self._nwb_filepath

    def write_nwb(self, **kwargs):
        """Tries to write nwb to disk. If it fails, the filepath has ".error"
        appended

        Parameters
        ----------
        kwargs: kwargs sent to `from_nwb`, `to_nwb`

        """
        from_lims_kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(self._serializer.from_lims).parameters}
        lims_session = self._serializer.from_lims(
            behavior_session_id=self._session_data['behavior_session_id'],
            **from_lims_kwargs)

        try:
            nwbfile = self._write_nwb(
                session=lims_session, **kwargs)
            self._compare_sessions(nwbfile=nwbfile, lims_session=lims_session,
                                   **kwargs)
            os.rename(self.nwb_filepath_inprogress, self._nwb_filepath)
        except Exception as e:
            if os.path.isfile(self.nwb_filepath_inprogress):
                os.rename(self.nwb_filepath_inprogress,
                          self._nwb_filepath_error)
            raise e

    def _write_nwb(
            self,
            session: BehaviorSession,
            **kwargs) -> NWBFile:
        """

        Parameters
        ----------
        session_data
        kwargs: kwargs to pass to `to_nwb`

        Returns
        -------

        """
        to_nwb_kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(self._serializer.to_nwb).parameters}
        nwbfile = session.to_nwb(**to_nwb_kwargs)

        with NWBHDF5IO(self.nwb_filepath_inprogress, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)
        return nwbfile

    def _compare_sessions(self, nwbfile: NWBFile, lims_session: DataObject,
                          **kwargs):
        kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(self._serializer.from_nwb).parameters}
        nwb_session = self._serializer.from_nwb(nwbfile, **kwargs)
        assert sessions_are_equal(lims_session, nwb_session, reraise=True)

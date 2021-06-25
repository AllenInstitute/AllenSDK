import numpy as np
import pandas as pd
from pynwb import NWBFile, ProcessingModule
from pynwb.ophys import OpticalChannel, ImageSegmentation

import allensdk.brain_observatory.roi_masks as roi
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.equipment import \
    EquipmentType
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_ophys_metadata import \
    BehaviorOphysMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.field_of_view_shape import \
    FieldOfViewShape
from allensdk.brain_observatory.nwb import CELL_SPECIMEN_COL_DESCRIPTIONS
from allensdk.internal.api import PostgresQueryMixin


class CellSpecimenTable(DataObject, LimsReadableInterface,
                        JsonReadableInterface, NwbReadableInterface,
                        NwbWritableInterface):
    def __init__(self, cell_specimen_table: pd.DataFrame):
        super().__init__(name='cell_specimen_table', value=cell_specimen_table)

    @classmethod
    def from_lims(cls, ophys_experiment_id: int,
                  lims_db: PostgresQueryMixin) -> "CellSpecimenTable":
        def get_ophys_cell_segmentation_run_id() -> int:
            """Get the ophys cell segmentation run id associated with an
            ophys experiment id"""
            query = """
                    SELECT oseg.id
                    FROM ophys_experiments oe
                    JOIN ophys_cell_segmentation_runs oseg
                    ON oe.id = oseg.ophys_experiment_id
                    WHERE oseg.current = 't'
                    AND oe.id = {};
                    """.format(ophys_experiment_id)
            return lims_db.fetchone(query, strict=True)

        ophys_cell_seg_run_id = get_ophys_cell_segmentation_run_id()
        query = """
                SELECT *
                FROM cell_rois cr
                WHERE cr.ophys_cell_segmentation_run_id = {};
                """.format(ophys_cell_seg_run_id)
        initial_cs_table = pd.read_sql(query, lims_db.get_connection())
        cell_specimen_table = initial_cs_table.rename(
            columns={'id': 'cell_roi_id', 'mask_matrix': 'roi_mask'})
        cell_specimen_table.drop(['ophys_experiment_id',
                                  'ophys_cell_segmentation_run_id'],
                                 inplace=True, axis=1)
        cell_specimen_table = cell_specimen_table.to_dict()
        fov_shape = FieldOfViewShape.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        cell_specimen_table = cls._postprocess(
            cell_specimen_table=cell_specimen_table, fov_shape=fov_shape)
        return cls(cell_specimen_table=cell_specimen_table)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "CellSpecimenTable":
        cell_specimen_table = dict_repr['cell_specimen_table_dict']
        fov_shape = FieldOfViewShape.from_json(dict_repr=dict_repr)
        cell_specimen_table = cls._postprocess(
            cell_specimen_table=cell_specimen_table, fov_shape=fov_shape)
        return cls(cell_specimen_table=cell_specimen_table)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile,
                 filter_invalid_rois=False) -> "CellSpecimenTable":
        # NOTE: ROI masks are stored in full frame width and height arrays
        df = nwbfile.processing[
            'ophys'].data_interfaces[
            'image_segmentation'].plane_segmentations[
            'cell_specimen_table'].to_dataframe()

        # Because pynwb stores this field as "image_mask", it is renamed here
        df = df.rename(columns={'image_mask': 'roi_mask'})

        df.index.rename('cell_roi_id', inplace=True)
        df['cell_specimen_id'] = [None if csid == -1 else csid
                                  for csid in df['cell_specimen_id'].values]

        df.reset_index(inplace=True)
        df.set_index('cell_specimen_id', inplace=True)

        if filter_invalid_rois:
            df = df[df["valid_roi"]]
        return cls(cell_specimen_table=df)

    def to_nwb(self, nwbfile: NWBFile, meta: BehaviorOphysMetadata) -> NWBFile:
        """
        :param nwbfile
            In-memory nwb file object
        :param meta
            Additional metadata not written to nwb yet, required to write
            cell specimen table
        """
        cell_roi_table = self.value.reset_index().set_index(
            'cell_roi_id')
        metadata = nwbfile.lab_meta_data['metadata']
        imaging_plane_meta = meta.ophys_metadata.imaging_plane

        # Device:
        equipment = meta.behavior_metadata.equipment
        if equipment.type == EquipmentType.MESOSCOPE:
            device_config = {
                "name": equipment.value,
                "description": "Allen Brain Observatory - Mesoscope 2P Rig"
            }
        else:
            device_config = {
                "name": equipment.value,
                "description": "Allen Brain Observatory - Scientifica 2P Rig",
                "manufacturer": "Scientifica"
            }
        nwbfile.create_device(**device_config)
        device = nwbfile.get_device(equipment.value)

        # FOV:
        fov_width = metadata.field_of_view_width
        fov_height = metadata.field_of_view_height
        imaging_plane_description = "{} field of view in {} at depth {} " \
                                    "um".format(
                                        (fov_width, fov_height),
                                        imaging_plane_meta.targeted_structure,
                                        metadata.imaging_depth)

        # Optical Channel:
        optical_channel = OpticalChannel(
            name='channel_1',
            description='2P Optical Channel',
            emission_lambda=meta.ophys_metadata.emission_lambda)

        # Imaging Plane:
        imaging_plane = nwbfile.create_imaging_plane(
            name='imaging_plane_1',
            optical_channel=optical_channel,
            description=imaging_plane_description,
            device=device,
            excitation_lambda=imaging_plane_meta.excitation_lambda,
            imaging_rate=imaging_plane_meta.ophys_frame_rate,
            indicator=meta.behavior_metadata.subject_metadata.indicator,
            location=imaging_plane_meta.targeted_structure)

        # Image Segmentation:
        image_segmentation = ImageSegmentation(name="image_segmentation")

        if 'ophys' not in nwbfile.processing:
            ophys_module = ProcessingModule('ophys', 'Ophys processing module')
            nwbfile.add_processing_module(ophys_module)
        else:
            ophys_module = nwbfile.processing['ophys']

        ophys_module.add_data_interface(image_segmentation)

        # Plane Segmentation:
        plane_segmentation = image_segmentation.create_plane_segmentation(
            name='cell_specimen_table',
            description="Segmented rois",
            imaging_plane=imaging_plane)

        for col_name in cell_roi_table.columns:
            # the columns 'roi_mask', 'pixel_mask', and 'voxel_mask' are
            # already defined in the nwb.ophys::PlaneSegmentation Object
            if col_name not in ['id', 'mask_matrix', 'roi_mask',
                                'pixel_mask', 'voxel_mask']:
                # This builds the columns with name of column and description
                # of column both equal to the column name in the cell_roi_table
                plane_segmentation.add_column(
                    col_name,
                    CELL_SPECIMEN_COL_DESCRIPTIONS.get(
                        col_name,
                        "No Description Available"))

        # go through each roi and add it to the plan segmentation object
        for cell_roi_id, table_row in cell_roi_table.iterrows():
            # NOTE: The 'roi_mask' in this cell_roi_table has already been
            # processing by the function from
            # allensdk.brain_observatory.behavior.session_apis.data_io
            # .ophys_lims_api
            # get_cell_specimen_table() method. As a result, the ROI is
            # stored in
            # an array that is the same shape as the FULL field of view of the
            # experiment (e.g. 512 x 512).
            mask = table_row.pop('roi_mask')

            csid = table_row.pop('cell_specimen_id')
            table_row['cell_specimen_id'] = -1 if csid is None else csid
            table_row['id'] = cell_roi_id
            plane_segmentation.add_roi(image_mask=mask, **table_row.to_dict())

        return nwbfile

    @staticmethod
    def _postprocess(cell_specimen_table: dict,
                     fov_shape: FieldOfViewShape) -> pd.DataFrame:
        """Converts raw cell_specimen_table dict to dataframe"""
        cell_specimen_table = pd.DataFrame.from_dict(
            cell_specimen_table).set_index(
            'cell_roi_id').sort_index()
        fov_width = fov_shape.width
        fov_height = fov_shape.height

        # Convert cropped ROI masks to uncropped versions
        roi_mask_list = []
        for cell_roi_id, table_row in cell_specimen_table.iterrows():
            # Deserialize roi data into AllenSDK RoiMask object
            curr_roi = roi.RoiMask(image_w=fov_width, image_h=fov_height,
                                   label=None, mask_group=-1)
            curr_roi.x = table_row['x']
            curr_roi.y = table_row['y']
            curr_roi.width = table_row['width']
            curr_roi.height = table_row['height']
            curr_roi.mask = np.array(table_row['roi_mask'])
            roi_mask_list.append(curr_roi.get_mask_plane().astype(np.bool))

        cell_specimen_table['roi_mask'] = roi_mask_list
        cell_specimen_table = cell_specimen_table[
            sorted(cell_specimen_table.columns)]

        cell_specimen_table.index.rename('cell_roi_id', inplace=True)
        cell_specimen_table.reset_index(inplace=True)
        cell_specimen_table.set_index('cell_specimen_id', inplace=True)
        return cell_specimen_table
